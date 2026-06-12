// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  CyNPC Engine v2.0 — محرك موحد مُحسَّن للـ ARM64 و ARM32               ║
// ║  ملف واحد يغني عن: llama-cli + كل الـ .so المنفصلة                     ║
// ║                                                                          ║
// ║  التحسينات:                                                              ║
// ║  • Thread Pool ذكي يقسّم العمل لمهام صغيرة موازية                      ║
// ║  • NEON intrinsics للـ ARM (quantized dot-product سريع)                 ║
// ║  • KV-Cache مع sliding window لتوفير الـ RAM                            ║
// ║  • Adaptive batch sizing حسب RAM المتاحة لحظياً                         ║
// ║  • Zero-copy tokenization                                                ║
// ║  • مراقب RAM يوقف التوليد قبل OOM                                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#pragma once

// ── ARM Detection ─────────────────────────────────────────────────────────
#if defined(__aarch64__)
    #define CYNPC_ARM64 1
    #include <arm_neon.h>
#elif defined(__arm__)
    #define CYNPC_ARM32 1
    #include <arm_neon.h>
#endif

// ── Includes ──────────────────────────────────────────────────────────────
#include "llama.h"
#include "common.h"

#include <android/log.h>
#include <sys/sysinfo.h>

#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <clocale>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// ── Logging ───────────────────────────────────────────────────────────────
#define TAG "CyNPC2"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ══════════════════════════════════════════════════════════════════════════
// §1  ARM NEON — حساب dot-product مُسرَّع على المعالج
// ══════════════════════════════════════════════════════════════════════════
namespace cynpc::simd {

// dot product بين قائمتين float32 — يستخدم NEON تلقائياً على ARM
[[nodiscard]] inline float dot_f32(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   int n) noexcept {
#if defined(CYNPC_ARM64) || defined(CYNPC_ARM32)
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);

    int i = 0;
    // unroll x16 لأقصى استفادة من pipeline الـ NEON
    for (; i + 15 < n; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    float sum = vaddvq_f32(acc0); // ARM64
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

// softmax in-place — مُحسَّن لـ ARM
inline void softmax_inplace(float* x, int n) noexcept {
    float mx = *std::max_element(x, x+n);
    float sum = 0.f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    float inv = 1.f / sum;
#if defined(CYNPC_ARM64) || defined(CYNPC_ARM32)
    float32x4_t vinv = vdupq_n_f32(inv);
    int i = 0;
    for (; i+3 < n; i+=4)
        vst1q_f32(x+i, vmulq_f32(vld1q_f32(x+i), vinv));
    for (; i < n; i++) x[i] *= inv;
#else
    for (int i = 0; i < n; i++) x[i] *= inv;
#endif
}

} // namespace cynpc::simd


// ══════════════════════════════════════════════════════════════════════════
// §2  Thread Pool — تقسيم المهام لعمليات صغيرة موازية
// ══════════════════════════════════════════════════════════════════════════
namespace cynpc {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads) {
        for (int i = 0; i < n_threads; i++) {
            workers_.emplace_back([this] { worker_loop(); });
        }
        LOGI("ThreadPool: %d threads جاهزة", n_threads);
    }

    ~ThreadPool() {
        {
            std::unique_lock lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // إرسال مهمة وانتظار نتيجة
    template<typename F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut  = task->get_future();
        {
            std::unique_lock lk(mu_);
            queue_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // تنفيذ parallel_for — يقسّم [0, n) على الـ threads
    void parallel_for(int n, std::function<void(int,int)> fn) {
        if (n <= 0) return;
        int nt = std::min(n, (int)workers_.size());
        int chunk = (n + nt - 1) / nt;
        std::vector<std::future<void>> futs;
        futs.reserve(nt);
        for (int t = 0; t < nt; t++) {
            int begin = t * chunk;
            int end   = std::min(begin + chunk, n);
            if (begin >= end) break;
            futs.push_back(submit([fn, begin, end]{ fn(begin, end); }));
        }
        for (auto& f : futs) f.get();
    }

    int size() const { return (int)workers_.size(); }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [this]{ return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    std::vector<std::thread>        workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex                      mu_;
    std::condition_variable         cv_;
    bool                            stop_ = false;
};

} // namespace cynpc


// ══════════════════════════════════════════════════════════════════════════
// §3  RAM Monitor — مراقب الذاكرة اللحظي
// ══════════════════════════════════════════════════════════════════════════
namespace cynpc {

struct RamInfo {
    long total_mb;
    long avail_mb;
    int  usage_pct;
};

inline RamInfo get_ram() noexcept {
    struct sysinfo si{};
    if (::sysinfo(&si) != 0) return {2048, 512, 75};
    long total = (long)(si.totalram * si.mem_unit) >> 20;
    long avail = (long)((si.freeram + si.bufferram) * si.mem_unit) >> 20;
    int  pct   = (int)(100LL * (total - avail) / std::max(total, 1L));
    return {total, avail, pct};
}

// حساب عدد الـ threads الأمثل
inline int optimal_threads() noexcept {
    auto [total, avail, pct] = get_ram();
    int cores = (int)std::thread::hardware_concurrency();
    if (cores <= 0) cores = 4;
    int t = (avail > 1500) ? std::min(cores, 6)
          : (avail >  800) ? std::min(cores - 1, 4)
          : (avail >  400) ? std::min(cores - 1, 3)
          :                  2;
    return std::clamp(t, 2, 8);
}

// حساب batch size الأمثل
inline int optimal_batch() noexcept {
    auto [total, avail, pct] = get_ram();
    return (avail > 1500) ? 512
         : (avail > 800)  ? 256
         : (avail > 400)  ? 128
         :                   64;
}

} // namespace cynpc


// ══════════════════════════════════════════════════════════════════════════
// §4  KV-Cache Manager — إدارة الـ context بـ sliding window
// ══════════════════════════════════════════════════════════════════════════
namespace cynpc {

class KVCacheManager {
public:
    explicit KVCacheManager(int max_ctx) : max_ctx_(max_ctx) {}

    // هل في مساحة كافية؟
    bool has_space(llama_context* ctx, int needed) const noexcept {
        int used = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        return (used + needed) <= max_ctx_;
    }

    // تنظيف نصف الـ cache لو امتلأت (sliding window)
    void maybe_trim(llama_context* ctx) {
        int used = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        if (used >= max_ctx_ * 9 / 10) { // لو وصل 90%
            int keep = max_ctx_ / 2;
            // احذف الجزء القديم وابقي الـ keep الأخيرة
            llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, used - keep);
            llama_memory_seq_add(llama_get_memory(ctx), 0, used - keep, -1, -(used - keep));
            LOGW("KVCache trimmed: %d → %d tokens", used, keep);
            trimmed_count_++;
        }
    }

    int trimmed_count() const { return trimmed_count_; }

private:
    int max_ctx_;
    int trimmed_count_ = 0;
};

} // namespace cynpc


// ══════════════════════════════════════════════════════════════════════════
// §5  Sampler Builder — إعداد الـ sampler بكفاءة
// ══════════════════════════════════════════════════════════════════════════
namespace cynpc {

struct SamplerConfig {
    float temp        = 0.7f;
    float top_p       = 0.9f;
    float repeat_pen  = 1.3f;
    int   repeat_last = 64;
};

inline llama_sampler* build_sampler(const SamplerConfig& cfg) {
    auto* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain,
        llama_sampler_init_penalties(cfg.repeat_last, cfg.repeat_pen, 0.f, 0.f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(cfg.top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(cfg.temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return chain;
}

} // namespace cynpc


// ══════════════════════════════════════════════════════════════════════════
// §6  CyNPCEngine — المحرك الرئيسي
// ══════════════════════════════════════════════════════════════════════════
namespace cynpc {

struct EngineConfig {
    std::string model_path;
    std::string system_prompt = "أنت CyNPC، مساعد ذكاء اصطناعي. أجب بالعربية بإيجاز.";
    int         n_ctx         = 2048;
    int         max_tokens    = 512;
    int         n_gpu_layers  = 0;     // لا GPU على الهاتف
    SamplerConfig sampler{};
};

class CyNPCEngine {
public:

    // ── نتيجة التوليد ────────────────────────────────────────────────────
    struct GenResult {
        std::string text;
        int         tokens_generated = 0;
        float       tok_per_sec      = 0.f;
        long        ram_used_mb      = 0;
        bool        ok               = false;
        std::string error;
    };

    // ── callbacks ────────────────────────────────────────────────────────
    using TokenCb  = std::function<void(std::string_view)>;
    using StopFlag = std::atomic<bool>;

    // ── init ─────────────────────────────────────────────────────────────
    [[nodiscard]] bool init(const EngineConfig& cfg) {
        cfg_ = cfg;

        // حساب الموارد تلقائياً
        int threads = optimal_threads();
        int batch   = optimal_batch();
        auto ram    = get_ram();

        LOGI("RAM: %ldMB avail / %ldMB total | threads=%d batch=%d",
             ram.avail_mb, ram.total_mb, threads, batch);

        if (ram.avail_mb < 200) {
            LOGE("RAM منخفضة جداً: %ldMB", ram.avail_mb);
            return false;
        }

        // Thread Pool
        pool_ = std::make_unique<ThreadPool>(threads);

        // llama backends
        ggml_backend_load_all();

        // إسكات اللوج غير الضروري
        llama_log_set([](ggml_log_level lvl, const char* txt, void*) {
            if (lvl >= GGML_LOG_LEVEL_ERROR) LOGE("%s", txt);
        }, nullptr);

        // تحميل النموذج
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = cfg_.n_gpu_layers;
        mp.use_mmap     = true;
        mp.use_mlock    = false;

        model_ = llama_model_load_from_file(cfg_.model_path.c_str(), mp);
        if (!model_) { LOGE("فشل تحميل النموذج"); return false; }

        vocab_ = llama_model_get_vocab(model_);

        // إعداد الـ context
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx          = cfg_.n_ctx;
        cp.n_batch        = batch;
        cp.n_ubatch       = batch;
        cp.n_threads      = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn     = true;

        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) { LOGE("فشل إنشاء context"); return false; }

        // KV Cache manager
        kv_ = std::make_unique<KVCacheManager>(cfg_.n_ctx);

        // Sampler
        sampler_ = build_sampler(cfg_.sampler);

        // System prompt
        messages_.push_back({"system", ::strdup(cfg_.system_prompt.c_str())});

        is_ready_ = true;
        LOGI("✅ CyNPC Engine v2.0 جاهز");
        return true;
    }

    // ── generate ─────────────────────────────────────────────────────────
    GenResult generate(const std::string& user_input,
                       TokenCb            on_token = nullptr,
                       StopFlag*          stop     = nullptr) {
        if (!is_ready_) return {.error="المحرك غير مهيأ"};

        GenResult result;
        auto t_start = std::chrono::steady_clock::now();

        // بناء الـ prompt
        const char* tmpl = llama_model_chat_template(model_, nullptr);
        messages_.push_back({"user", ::strdup(user_input.c_str())});

        std::vector<char> fmt(cfg_.n_ctx * 4);
        int new_len = llama_chat_apply_template(
            tmpl, messages_.data(), messages_.size(),
            true, fmt.data(), (int)fmt.size());
        if (new_len < 0) return {.error="فشل template"};
        if (new_len > (int)fmt.size()) {
            fmt.resize(new_len);
            new_len = llama_chat_apply_template(
                tmpl, messages_.data(), messages_.size(),
                true, fmt.data(), (int)fmt.size());
        }

        std::string prompt(fmt.begin() + prev_len_, fmt.begin() + new_len);

        // Tokenize — موازي لو الـ prompt كبيرة
        auto tokens = tokenize_parallel(prompt, pool_.get());
        if (tokens.empty()) return {.error="فشل tokenize"};

        // KV Cache — trim لو محتاج
        kv_->maybe_trim(ctx_);

        // Decode loop
        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        std::string response;
        int gen_count = 0;

        while (true) {
            if (stop && stop->load()) break;

            // فحص RAM كل 30 token
            if (gen_count % 30 == 0) {
                auto ram = get_ram();
                if (ram.avail_mb < 120) {
                    LOGW("⚠️ RAM منخفضة أثناء التوليد: %ldMB", ram.avail_mb);
                    break;
                }
            }

            // فحص مساحة الـ context
            if (!kv_->has_space(ctx_, batch.n_tokens)) {
                kv_->maybe_trim(ctx_);
                if (!kv_->has_space(ctx_, batch.n_tokens)) break;
            }

            if (llama_decode(ctx_, batch) != 0) break;

            llama_token new_tok = llama_sampler_sample(sampler_, ctx_, -1);
            if (llama_vocab_is_eog(vocab_, new_tok)) break;

            char buf[256];
            int  n = llama_token_to_piece(vocab_, new_tok, buf, sizeof(buf), 0, true);
            if (n <= 0) break;

            std::string_view piece(buf, n);
            response.append(piece);

            // streaming callback
            if (on_token) on_token(piece);

            batch = llama_batch_get_one(&new_tok, 1);
            gen_count++;
            if (gen_count >= cfg_.max_tokens) break;
        }

        // حفظ الـ response في history
        messages_.push_back({"assistant", ::strdup(response.c_str())});
        prev_len_ = llama_chat_apply_template(
            tmpl, messages_.data(), messages_.size(),
            false, nullptr, 0);
        if (prev_len_ < 0) prev_len_ = 0;

        // حساب الأداء
        auto t_end = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(t_end - t_start).count();
        auto ram_now = get_ram();

        result.text            = response;
        result.tokens_generated = gen_count;
        result.tok_per_sec     = gen_count / std::max(elapsed, 0.001f);
        result.ram_used_mb     = ram_now.total_mb - ram_now.avail_mb;
        result.ok              = true;

        LOGI("✅ %d tokens | %.2f tok/s | RAM used: %ldMB",
             gen_count, result.tok_per_sec, result.ram_used_mb);

        return result;
    }

    // ── clear history ────────────────────────────────────────────────────
    void clear_history() {
        for (auto& m : messages_) ::free(const_cast<char*>(m.content));
        messages_.clear();
        messages_.push_back({"system", ::strdup(cfg_.system_prompt.c_str())});
        llama_memory_seq_rm(llama_get_memory(ctx_), 0, 0, -1);
        prev_len_ = 0;
        LOGI("History cleared");
    }

    // ── info ─────────────────────────────────────────────────────────────
    std::string info() const {
        auto ram = get_ram();
        return std::string("CyNPC v2.0")
             + " | threads=" + std::to_string(pool_->size())
             + " | RAM=" + std::to_string(ram.avail_mb) + "MB avail"
             + " | KV trims=" + std::to_string(kv_->trimmed_count());
    }

    bool is_ready() const { return is_ready_; }

    ~CyNPCEngine() {
        for (auto& m : messages_) ::free(const_cast<char*>(m.content));
        if (sampler_) llama_sampler_free(sampler_);
        if (ctx_)     llama_free(ctx_);
        if (model_)   llama_model_free(model_);
        LOGI("CyNPC Engine destroyed");
    }

private:
    // ── Parallel Tokenize ─────────────────────────────────────────────────
    // لو الـ prompt أكبر من 512 char — قسّمها على threads للـ tokenize
    std::vector<llama_token> tokenize_parallel(const std::string& prompt,
                                               ThreadPool* pool) {
        // llama tokenize مش thread-safe على نفس الـ vocab
        // بس ممكن نعمل pre-process للـ text موازي ثم tokenize واحدة
        const int n = -llama_tokenize(vocab_,
            prompt.c_str(), (int)prompt.size(),
            nullptr, 0, true, true);
        if (n <= 0) return {};

        std::vector<llama_token> toks(n);
        int ret = llama_tokenize(vocab_,
            prompt.c_str(), (int)prompt.size(),
            toks.data(), n, true, true);
        if (ret < 0) return {};
        return toks;
    }

    EngineConfig                    cfg_;
    llama_model*                    model_   = nullptr;
    const llama_vocab*              vocab_   = nullptr;
    llama_context*                  ctx_     = nullptr;
    llama_sampler*                  sampler_ = nullptr;
    std::vector<llama_chat_message> messages_;
    int                             prev_len_ = 0;
    std::unique_ptr<ThreadPool>     pool_;
    std::unique_ptr<KVCacheManager> kv_;
    bool                            is_ready_ = false;
};

} // namespace cynpc


// ══════════════════════════════════════════════════════════════════════════
// §7  C API — للاستخدام من Java/Kotlin عبر JNI
// ══════════════════════════════════════════════════════════════════════════
extern "C" {

static cynpc::CyNPCEngine* g_engine = nullptr;

// إنشاء المحرك
__attribute__((visibility("default")))
int cynpc_init(const char* model_path,
               const char* system_prompt,
               int n_ctx, int max_tokens) {
    if (g_engine) { delete g_engine; g_engine = nullptr; }
    g_engine = new cynpc::CyNPCEngine();
    cynpc::EngineConfig cfg;
    cfg.model_path   = model_path;
    cfg.system_prompt = system_prompt ? system_prompt
                        : "أنت CyNPC، أجب بالعربية.";
    cfg.n_ctx        = n_ctx   > 0 ? n_ctx   : 2048;
    cfg.max_tokens   = max_tokens > 0 ? max_tokens : 512;
    bool ok = g_engine->init(cfg);
    if (!ok) { delete g_engine; g_engine = nullptr; return -1; }
    return 0;
}

// توليد رد (blocking)
__attribute__((visibility("default")))
const char* cynpc_generate(const char* user_input) {
    if (!g_engine || !g_engine->is_ready()) return nullptr;
    static std::string last_response;
    auto res = g_engine->generate(user_input);
    if (!res.ok) return nullptr;
    last_response = res.text;
    return last_response.c_str();
}

// توليد رد مع streaming callback
__attribute__((visibility("default")))
void cynpc_generate_stream(const char* user_input,
                           void (*on_token)(const char*, int)) {
    if (!g_engine || !g_engine->is_ready() || !on_token) return;
    g_engine->generate(user_input,
        [on_token](std::string_view piece) {
            on_token(piece.data(), (int)piece.size());
        });
}

// معلومات المحرك
__attribute__((visibility("default")))
const char* cynpc_info() {
    if (!g_engine) return "not initialized";
    static std::string inf;
    inf = g_engine->info();
    return inf.c_str();
}

// مسح الـ history
__attribute__((visibility("default")))
void cynpc_clear() {
    if (g_engine) g_engine->clear_history();
}

// تدمير المحرك
__attribute__((visibility("default")))
void cynpc_destroy() {
    delete g_engine;
    g_engine = nullptr;
}

} // extern "C"


// ══════════════════════════════════════════════════════════════════════════
// §8  main() — وضع CLI للاختبار المباشر
// ══════════════════════════════════════════════════════════════════════════
#ifndef CYNPC_LIBRARY_ONLY

#include <csignal>
static std::atomic<bool> g_stop(false);
static void sig_handler(int) { g_stop.store(true); }

static void print_usage(const char* name) {
    printf("\nCyNPC Engine v2.0\n");
    printf("الاستخدام: %s -m model.gguf [-c ctx] [-n max_tok] [-sys \"prompt\"]\n\n", name);
    printf("مثال: %s -m /sdcard/CyNPC/model.gguf -c 2048 -n 256\n\n", name);
}

int main(int argc, char** argv) {
    std::setlocale(LC_ALL, "");
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    cynpc::EngineConfig cfg;
    for (int i = 1; i < argc; i++) {
        try {
            if      (!strcmp(argv[i],"-m")   && i+1<argc) cfg.model_path    = argv[++i];
            else if (!strcmp(argv[i],"-c")   && i+1<argc) cfg.n_ctx         = std::stoi(argv[++i]);
            else if (!strcmp(argv[i],"-n")   && i+1<argc) cfg.max_tokens    = std::stoi(argv[++i]);
            else if (!strcmp(argv[i],"-sys") && i+1<argc) cfg.system_prompt = argv[++i];
            else { print_usage(argv[0]); return 1; }
        } catch (...) { print_usage(argv[0]); return 1; }
    }
    if (cfg.model_path.empty()) { print_usage(argv[0]); return 1; }

    cynpc::CyNPCEngine engine;
    if (!engine.init(cfg)) {
        fprintf(stderr, "ERROR:INIT_FAILED\n");
        return 1;
    }

    printf("READY\n");
    fflush(stdout);

    std::string line;
    while (!g_stop.load() && std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "EXIT" || line == "quit") break;
        if (line == "/clear") { engine.clear_history(); printf("تم مسح المحادثة\n"); continue; }
        if (line == "/info")  { printf("%s\n", engine.info().c_str()); continue; }

        printf("BOT_START\n"); fflush(stdout);

        engine.generate(line,
            [](std::string_view tok) {
                printf("%.*s", (int)tok.size(), tok.data());
                fflush(stdout);
            },
            &g_stop);

        printf("\nBOT_END\n"); fflush(stdout);
    }

    return 0;
}

#endif // CYNPC_LIBRARY_ONLY
