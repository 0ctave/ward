/* pam_ward.so — native PAM module for ward (DESIGN.md §9b, roadmap K1).
 *
 * Why a native module: pam_exec can forward our messages to the greeter, but it cannot hand
 * PAM an authentication token. pam_gnome_keyring / pam_kwallet5 unlock the login keyring with
 * that token, so after a face login the keyring stayed locked. This module runs the same
 * helper (/usr/local/bin/ward-auth), forwards its stdout lines as PAM_TEXT_INFO messages,
 * and on an accept — if the user enrolled a sealed credential with `ward keyring enable` —
 * unseals it with systemd-creds (TPM2-bound where available) and calls
 * pam_set_item(PAM_AUTHTOK), exactly as if the password had been typed.
 *
 * Fail-open by construction: every path other than a helper exit status of 0 returns
 * PAM_AUTH_ERR, which the stack treats like a wrong face → next module (the password).
 * The module never returns PAM_SUCCESS without the helper saying so, never prompts, and
 * never touches the stack's other items.
 *
 * Built without libpam headers on purpose (the -dev package is often missing): the handful of
 * libpam entry points used are declared here with the ABI Linux-PAM has had since 1.0. The
 * symbols resolve from the libpam already loaded in the calling process.
 *
 * Options (all optional):   helper=/path/to/ward-auth   creds_dir=/etc/ward/keyring
 *                           creds=/usr/bin/systemd-creds   timeout=6   debug
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

/* ---- the subset of <security/pam_modules.h> / <security/pam_appl.h> we use ---- */
typedef struct pam_handle pam_handle_t;
struct pam_message { int msg_style; const char *msg; };
struct pam_response { char *resp; int resp_retcode; };
struct pam_conv { int (*conv)(int, const struct pam_message **, struct pam_response **, void *); void *appdata_ptr; };
#define PAM_SUCCESS 0
#define PAM_AUTH_ERR 7
#define PAM_IGNORE 25
#define PAM_SERVICE 1
#define PAM_USER 2
#define PAM_TTY 3
#define PAM_RHOST 4
#define PAM_CONV 5
#define PAM_AUTHTOK 6
#define PAM_TEXT_INFO 4
extern int pam_get_user(pam_handle_t *pamh, const char **user, const char *prompt);
extern int pam_get_item(const pam_handle_t *pamh, int item_type, const void **item);
extern int pam_set_item(pam_handle_t *pamh, int item_type, const void *item);

#define DEFAULT_HELPER "/usr/local/bin/ward-auth"
#define DEFAULT_CREDS_DIR "/etc/ward/keyring"
#define DEFAULT_CREDS "/usr/bin/systemd-creds"

struct opts { const char *helper, *creds_dir, *creds; int timeout, debug; };

static void logmsg(int prio, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    openlog("pam_ward", LOG_PID, LOG_AUTHPRIV);
    vsyslog(prio, fmt, ap);
    closelog();
    va_end(ap);
}

static void info(pam_handle_t *pamh, const char *text) {
    const struct pam_conv *conv = NULL;
    if (pam_get_item(pamh, PAM_CONV, (const void **)&conv) != PAM_SUCCESS || conv == NULL || conv->conv == NULL)
        return;
    struct pam_message m = { PAM_TEXT_INFO, text };
    const struct pam_message *mp = &m;
    struct pam_response *resp = NULL;
    if (conv->conv(1, &mp, &resp, conv->appdata_ptr) == PAM_SUCCESS && resp) {
        if (resp->resp) free(resp->resp);
        free(resp);
    }
}

static void parse_opts(struct opts *o, int argc, const char **argv) {
    /* 8 s default sits above the helper's own cascade (engine hard_timeout ~4 s < daemon read ~8 s
     * < ward-auth process backstop ~12 s, see ward/client.py); if hard_timeout_ms is raised past
     * ~4 s, pass timeout=N here to match. On timeout we fall through to the password (fail open). */
    o->helper = DEFAULT_HELPER; o->creds_dir = DEFAULT_CREDS_DIR; o->creds = DEFAULT_CREDS; o->timeout = 8; o->debug = 0;
    for (int i = 0; i < argc; i++) {
        if (!strncmp(argv[i], "helper=", 7)) o->helper = argv[i] + 7;
        else if (!strncmp(argv[i], "creds_dir=", 10)) o->creds_dir = argv[i] + 10;
        else if (!strncmp(argv[i], "creds=", 6)) o->creds = argv[i] + 6;
        else if (!strncmp(argv[i], "timeout=", 8)) o->timeout = atoi(argv[i] + 8);
        else if (!strcmp(argv[i], "debug")) o->debug = 1;
    }
    if (o->timeout < 1) o->timeout = 8;
}

static const char *item_str(pam_handle_t *pamh, int type) {
    const void *v = NULL;
    if (pam_get_item(pamh, type, &v) != PAM_SUCCESS || v == NULL) return "";
    return (const char *)v;
}

/* Run the helper as root with PAM_* in its environment; forward each stdout line as a PAM
 * info message; return its exit status (or -1). */
static int run_helper_(pam_handle_t *pamh, const struct opts *o, const char *user);
/* like pam_exec: the caller may ignore or reap SIGCHLD itself, which would make our waitpid
 * fail (ECHILD) and turn a real accept into a failure — default it for the helper's lifetime */
static int run_helper(pam_handle_t *pamh, const struct opts *o, const char *user) {
    struct sigaction dfl, old;
    memset(&dfl, 0, sizeof dfl); dfl.sa_handler = SIG_DFL; sigemptyset(&dfl.sa_mask);
    int saved = sigaction(SIGCHLD, &dfl, &old) == 0;
    int rc = run_helper_(pamh, o, user);
    if (saved) sigaction(SIGCHLD, &old, NULL);
    return rc;
}

static int run_helper_(pam_handle_t *pamh, const struct opts *o, const char *user) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return -1; }
    if (pid == 0) {
        setsid();                                               /* own group: a timeout kills it whole */
        close(fds[0]);
        int null = open("/dev/null", O_RDWR);
        if (null >= 0) { dup2(null, 0); dup2(null, 2); if (null > 2) close(null); }
        dup2(fds[1], 1); close(fds[1]);
        for (int fd = 3; fd < 256; fd++) close(fd);
        /* sudo runs PAM with real uid = the caller: make both uids root so the helper can read the store */
        if (geteuid() == 0 && (setgid(0) != 0 || setuid(0) != 0)) _exit(126);
        /* a root child must not inherit the caller's environment: through setuid sudo or the
         * polkit agent helper that is the unprivileged user's (PYTHONHOME, HOME → user site…) */
        clearenv();
        setenv("LANG", "C.UTF-8", 1);
        /* The PAM_* subset of the canonical forward list (ward/daemon.py PAM_ENV); keep in step. */
        setenv("PAM_USER", user, 1);
        setenv("PAM_TYPE", "auth", 1);
        setenv("PAM_SERVICE", item_str(pamh, PAM_SERVICE), 1);
        setenv("PAM_TTY", item_str(pamh, PAM_TTY), 1);
        setenv("PAM_RHOST", item_str(pamh, PAM_RHOST), 1);
        setenv("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", 1);
        execl(o->helper, o->helper, (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    /* Fail open on a wedged helper (camera stuck in the driver, a hung model load): poll the pipe
     * against a hard deadline, kill the whole process group when it passes, never block in
     * waitpid. A helper that prints nothing must time out too — fgets alone would wait forever. */
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    long deadline_ms = (long)o->timeout * 1000;
    char line[512]; size_t n = 0; int timed_out = 0;
    for (;;) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        long left = deadline_ms - ((now.tv_sec - t0.tv_sec) * 1000 + (now.tv_nsec - t0.tv_nsec) / 1000000);
        if (left <= 0) { timed_out = 1; break; }
        struct pollfd pfd = { fds[0], POLLIN, 0 };
        int pr = poll(&pfd, 1, (int)left);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) { timed_out = 1; break; }
        char c; ssize_t r = read(fds[0], &c, 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;                                          /* EOF: the helper finished */
        if (c == '\n' || n == sizeof line - 1) {
            line[n] = '\0';
            while (n && line[n - 1] == '\r') line[--n] = '\0';
            if (n) info(pamh, line);
            n = 0;
            if (c != '\n') line[n++] = c;
        } else line[n++] = c;
    }
    close(fds[0]);
    if (timed_out) {
        if (o->debug) logmsg(LOG_NOTICE, "helper timed out after %d s: falling through", o->timeout);
        kill(-pid, SIGKILL); kill(pid, SIGKILL);
    }
    int status = 0;
    for (int i = 0; i < 200; i++) {                                 /* reap, bounded to ~2 s */
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) break;
        if (w < 0) return -1;
        if (i == 199) {                                             /* still exiting: kill it whole, one last reap */
            kill(-pid, SIGKILL); kill(pid, SIGKILL);
            usleep(10000); waitpid(pid, NULL, WNOHANG);
            return -1;
        }
        usleep(10000);
    }
    if (timed_out || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

/* The keyring unseal, started in the BACKGROUND so its cost (a TPM2 decrypt can be 2–4 s on a
 * firmware TPM) overlaps face recognition instead of adding to it. `unseal_start` forks
 * systemd-creds and returns its pid + read fd; `unseal_finish` collects the plaintext. */
#define UNSEAL_TIMEOUT_MS 5000
/* once the face is accepted the login waits for the keyring at most this much longer: a slow or
 * failing decrypt (a TPM seal after a firmware update) must not stall an unlock that already passed */
#define UNSEAL_GRACE_MS 1000
struct unseal_job { pid_t pid; int fd; int active; struct timespec t0; };

static void unseal_start(const struct opts *o, const char *user, struct unseal_job *j) {
    j->pid = -1; j->fd = -1; j->active = 0;
    char path[512], name[160];
    if (strchr(user, '/') || user[0] == '.') return;
    snprintf(path, sizeof path, "%s/%s.cred", o->creds_dir, user);
    if (access(path, R_OK) != 0) return;                        /* no credential enrolled: fine */
    snprintf(name, sizeof name, "--name=ward-keyring-%s", user);
    int fds[2];
    if (pipe(fds) != 0) return;
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return; }
    if (pid == 0) {
        close(fds[0]);
        int null = open("/dev/null", O_RDWR);
        if (null >= 0) { dup2(null, 0); dup2(null, 2); if (null > 2) close(null); }
        dup2(fds[1], 1); close(fds[1]);
        for (int fd = 3; fd < 256; fd++) close(fd);
        if (geteuid() == 0 && (setgid(0) != 0 || setuid(0) != 0)) _exit(126);
        clearenv();                                             /* never the caller's environment as root */
        setenv("LANG", "C.UTF-8", 1);
        setenv("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", 1);
        execl(o->creds, o->creds, "decrypt", name, path, "-", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    j->pid = pid; j->fd = fds[0]; j->active = 1;
    clock_gettime(CLOCK_MONOTONIC, &j->t0);
}

static int unseal_finish(struct unseal_job *j, char *buf, size_t size) {
    if (!j->active) return 1;                                   /* nothing was started */
    /* bounded like the helper: a stuck TPM decrypt must not hang a login that the face already
     * passed — on timeout the keyring simply falls back to its own prompt */
    size_t got = 0; int timed_out = 0;
    struct timespec t0 = j->t0, accepted; clock_gettime(CLOCK_MONOTONIC, &accepted);
    long budget = (accepted.tv_sec - t0.tv_sec) * 1000 + (accepted.tv_nsec - t0.tv_nsec) / 1000000 + UNSEAL_GRACE_MS;
    if (budget > UNSEAL_TIMEOUT_MS) budget = UNSEAL_TIMEOUT_MS;
    for (;;) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        long left = budget - ((now.tv_sec - t0.tv_sec) * 1000 + (now.tv_nsec - t0.tv_nsec) / 1000000);
        if (left <= 0) { timed_out = 1; break; }
        struct pollfd pfd = { j->fd, POLLIN, 0 };
        int pr = poll(&pfd, 1, (int)left);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) { timed_out = 1; break; }
        ssize_t r = read(j->fd, buf + got, size - 1 - got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= size - 1) break;
    }
    close(j->fd);
    j->active = 0;
    int status = 0, reaped = 0;
    if (timed_out) kill(j->pid, SIGKILL);
    for (int i = 0; i < 200; i++) {
        pid_t w = waitpid(j->pid, &status, WNOHANG);
        if (w == j->pid) { reaped = 1; break; }
        if (w < 0) break;
        usleep(10000);
    }
    if (!reaped) {                                              /* never read an unreaped child as success */
        kill(j->pid, SIGKILL);
        for (int i = 0; i < 20 && waitpid(j->pid, NULL, WNOHANG) == 0; i++) usleep(10000);
        memset(buf, 0, size); return -1;
    }
    if (timed_out || got >= size - 1) { memset(buf, 0, size); return -1; }   /* a full buffer may be truncated */
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || got == 0) { memset(buf, 0, size); return -1; }
    buf[got] = '\0';
    while (got && (buf[got - 1] == '\n' || buf[got - 1] == '\r')) buf[--got] = '\0';
    return got ? 0 : -1;
}

static void unseal_abort(struct unseal_job *j) {
    if (!j->active) return;
    /* bounded: a decrypt wedged in the TPM driver must never hold back the password module */
    kill(j->pid, SIGKILL);
    close(j->fd);
    for (int i = 0; i < 50 && waitpid(j->pid, NULL, WNOHANG) == 0; i++) usleep(10000);
    j->active = 0;
}

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)flags;
    struct opts o; parse_opts(&o, argc, argv);
    const char *user = NULL;
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || user == NULL || !*user) return PAM_AUTH_ERR;
    if (access(o.helper, X_OK) != 0) { if (o.debug) logmsg(LOG_DEBUG, "helper %s missing", o.helper); return PAM_AUTH_ERR; }
    /* start the keyring decrypt now: it runs while the camera does its work, so a slow TPM is free */
    struct unseal_job job;
    unseal_start(&o, user, &job);
    int rc = run_helper(pamh, &o, user);
    if (o.debug) logmsg(LOG_DEBUG, "helper exit %d for %s", rc, user);
    if (rc != 0) { unseal_abort(&job); return PAM_AUTH_ERR; }
    char secret[1024];
    int u = unseal_finish(&job, secret, sizeof secret);
    if (u == 0) {
        if (pam_set_item(pamh, PAM_AUTHTOK, secret) != PAM_SUCCESS && o.debug) logmsg(LOG_DEBUG, "pam_set_item failed");
        memset(secret, 0, sizeof secret);
    } else if (u < 0) {
        logmsg(LOG_WARNING, "face accepted but the keyring credential for %s could not be unsealed", user);
    }
    return PAM_SUCCESS;
}

int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}
