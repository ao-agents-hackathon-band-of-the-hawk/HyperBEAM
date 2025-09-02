// --- FILE: native/wasi_nn_llama/src/wasi_nn_nif.c ---

#include "../include/wasi_nn_nif.h"
#include "../include/wasi_nn_logging.h"
#include <pthread.h>

#define LIB_PATH "./priv/libwasi_nn_backend.so"
#define MAX_MODEL_PATH 256
#define MAX_INPUT_SIZE 4096
#define MAX_CONFIG_SIZE 4096
#define MAX_OUTPUT_SIZE 8192
#define MAX_SESSION_ID_SIZE 256

// --- SINGLETON & RESOURCE DEFINITIONS ---

// Global singleton for the C++ backend, protected by a mutex
static void* g_backend_context = NULL;
static pthread_mutex_t g_backend_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_backend_initialized = 0;

// Resource to represent a lightweight session, not the whole backend
typedef struct {
    graph_execution_context exec_ctx;
} LlamaSession;

// Global struct to hold function pointers from the .so
static wasi_nn_backend_api g_wasi_nn_functions = {0};
// The resource type for our LlamaSession struct
static ErlNifResourceType* llama_session_resource;

// --- RESOURCE DESTRUCTOR ---

// This destructor is now SAFE. It only closes a session, it does not destroy the backend.
static void llama_session_destructor(ErlNifEnv* env, void* obj)
{
    LlamaSession* session = (LlamaSession*)obj;
    DRV_DEBUG("Destroying LlamaSession %p (exec_ctx: %u)\n", session, session->exec_ctx);

    pthread_mutex_lock(&g_backend_mutex);
    if (g_backend_initialized && session->exec_ctx != 0 && g_wasi_nn_functions.close_execution_context) {
        DRV_DEBUG("Closing execution context %u\n", session->exec_ctx);
        g_wasi_nn_functions.close_execution_context(g_backend_context, session->exec_ctx);
    }
    pthread_mutex_unlock(&g_backend_mutex);
    DRV_DEBUG("LlamaSession destroyed\n");
}

// --- NIF LIFECYCLE: load / unload ---

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    DRV_DEBUG("Load nif start\n");
    g_wasi_nn_functions.handle = dlopen(LIB_PATH, RTLD_LAZY);
    if (!g_wasi_nn_functions.handle) {
        DRV_DEBUG("Failed to load wasi library: %s\n", dlerror());
        return 1;
    }
    g_wasi_nn_functions.init_backend_with_config = (init_backend_with_config_fn)dlsym(g_wasi_nn_functions.handle, "init_backend_with_config");
    g_wasi_nn_functions.deinit_backend = (deinit_backend_fn)dlsym(g_wasi_nn_functions.handle, "deinit_backend");
    g_wasi_nn_functions.load_by_name_with_config = (load_by_name_with_config_fn)dlsym(g_wasi_nn_functions.handle, "load_by_name_with_config");
    g_wasi_nn_functions.init_execution_context = (init_execution_context_fn)dlsym(g_wasi_nn_functions.handle, "init_execution_context_with_session_id");
    g_wasi_nn_functions.close_execution_context = (close_execution_context_fn)dlsym(g_wasi_nn_functions.handle, "close_execution_context");
    g_wasi_nn_functions.run_inference = (run_inference_fn)dlsym(g_wasi_nn_functions.handle, "run_inference");

    if (!g_wasi_nn_functions.init_backend_with_config || !g_wasi_nn_functions.deinit_backend ||
        !g_wasi_nn_functions.load_by_name_with_config || !g_wasi_nn_functions.init_execution_context ||
        !g_wasi_nn_functions.close_execution_context || !g_wasi_nn_functions.run_inference) {
        DRV_DEBUG("Failed to load one or more required NIF functions.\n");
        dlclose(g_wasi_nn_functions.handle);
        return 1;
    }

    llama_session_resource = enif_open_resource_type(env, NULL, "llama_session",
        llama_session_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);

    if (llama_session_resource == NULL) {
        DRV_DEBUG("Failed to open llama_session_resource.\n");
        return 1;
    }
    DRV_DEBUG("Load nif Finished\n");
    return 0;
}

static void unload(ErlNifEnv* env, void* priv_data)
{
    DRV_DEBUG("Unloading NIF...\n");
    pthread_mutex_lock(&g_backend_mutex);
    if (g_backend_initialized) {
        DRV_DEBUG("Deinitializing global backend context.\n");
        g_wasi_nn_functions.deinit_backend(g_backend_context);
        g_backend_context = NULL;
        g_backend_initialized = 0;
    }
    pthread_mutex_unlock(&g_backend_mutex);

    if (g_wasi_nn_functions.handle) {
        dlclose(g_wasi_nn_functions.handle);
        g_wasi_nn_functions.handle = NULL;
    }
    DRV_DEBUG("NIF unloaded.\n");
}

// --- NIF FUNCTION IMPLEMENTATIONS ---

static ERL_NIF_TERM nif_load_by_name_with_config(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    char model_path[MAX_MODEL_PATH];
    char config[MAX_CONFIG_SIZE];

    if (argc != 2 ||
        !enif_get_string(env, argv[0], model_path, sizeof(model_path), ERL_NIF_LATIN1) ||
        !enif_get_string(env, argv[1], config, sizeof(config), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }

    pthread_mutex_lock(&g_backend_mutex);

    // Initialize the backend singleton on the first model load.
    if (!g_backend_initialized) {
        DRV_DEBUG("Global backend not initialized. Creating now...\n");
        wasi_nn_error err = g_wasi_nn_functions.init_backend_with_config(&g_backend_context, config, strlen(config));
        if (err != success) {
            pthread_mutex_unlock(&g_backend_mutex);
            return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "init_failed"));
        }
        g_backend_initialized = 1;
    }

    // Load the model. This will handle model switching safely inside the C++ layer.
    DRV_DEBUG("Loading model: %s\n", model_path);
    graph g; // A graph handle, not used beyond this scope in the new design.
    wasi_nn_error err = g_wasi_nn_functions.load_by_name_with_config(g_backend_context, model_path, strlen(model_path), config, strlen(config), &g);

    pthread_mutex_unlock(&g_backend_mutex);

    if (err != success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "load_failed"));
    }

    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_init_execution_context(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    char session_id[MAX_SESSION_ID_SIZE];
    if (argc != 1 || !enif_get_string(env, argv[0], session_id, sizeof(session_id), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }

    LlamaSession* session = enif_alloc_resource(llama_session_resource, sizeof(LlamaSession));
    if (!session) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "allocation_failed"));
    }
    session->exec_ctx = 0; // Initialize to a known invalid state.

    pthread_mutex_lock(&g_backend_mutex);
    if (!g_backend_initialized) {
        pthread_mutex_unlock(&g_backend_mutex);
        enif_release_resource(session);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "backend_not_initialized"));
    }

    wasi_nn_error err = g_wasi_nn_functions.init_execution_context(g_backend_context, session_id, &session->exec_ctx);
    pthread_mutex_unlock(&g_backend_mutex);

    if (err != success) {
        enif_release_resource(session);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "init_execution_failed"));
    }

    ERL_NIF_TERM session_term = enif_make_resource(env, session);
    enif_release_resource(session); // The Erlang term now owns the resource reference.

    return enif_make_tuple2(env, enif_make_atom(env, "ok"), session_term);
}

static ERL_NIF_TERM nif_run_inference(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaSession* session;
    char input[MAX_INPUT_SIZE];
    char options[MAX_CONFIG_SIZE];
    
    // Initialize options to an empty C string for the 2-arg case
    options[0] = '\0';

    if (argc < 2 || argc > 3 ||
        !enif_get_resource(env, argv[0], llama_session_resource, (void**)&session) ||
        !enif_get_string(env, argv[1], input, sizeof(input), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }

    if (argc == 3) {
        if (!enif_get_string(env, argv[2], options, sizeof(options), ERL_NIF_LATIN1)) {
            return enif_make_badarg(env);
        }
    }

    tensor_data output = (tensor_data)malloc(MAX_OUTPUT_SIZE);
    if (!output) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "allocation_failed"));
    }

    uint32_t output_size = MAX_OUTPUT_SIZE;
    tensor input_tensor = { .dimensions = NULL, .type = fp32, .data = (tensor_data)input };

    pthread_mutex_lock(&g_backend_mutex);
    if (!g_backend_initialized) {
        pthread_mutex_unlock(&g_backend_mutex);
        free(output);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "backend_not_initialized"));
    }

    wasi_nn_error err = g_wasi_nn_functions.run_inference(g_backend_context, session->exec_ctx, 0, &input_tensor, output, &output_size, options);
    pthread_mutex_unlock(&g_backend_mutex);

    if (err != success) {
        free(output);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "run_inference_failed"));
    }

    ERL_NIF_TERM result_bin;
    unsigned char* bin_data = enif_make_new_binary(env, output_size, &result_bin);
    if (!bin_data) {
        free(output);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "binary_creation_failed"));
    }
    memcpy(bin_data, output, output_size);
    free(output);

    return enif_make_tuple2(env, enif_make_atom(env, "ok"), result_bin);
}

// --- NIF FUNCTION TABLE ---

static ErlNifFunc nif_funcs[] = {
    {"nif_load_by_name_with_config", 2, nif_load_by_name_with_config, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_init_execution_context", 1, nif_init_execution_context},
    // Only export the 3-arity version. The 2-arity version is handled purely in Erlang.
    {"nif_run_inference", 3, nif_run_inference, ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, load, NULL, NULL, unload)