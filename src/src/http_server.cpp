// src/src/http_server.cpp
#include "ws_ai/http_server.h"   // 必须提供：class HttpServer { ... serve_forever(); ... }
#include "ws_ai/job_manager.h"   // 必须提供：JobManager
#include "ws_ai/config.h"        // 必须提供：Config

#include <httplib.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <memory>

namespace ws_ai {

// 你的 HttpServer 头文件里没有 job_manager_ 成员：这里用 cpp 内部静态指针兜底保存
static std::shared_ptr<JobManager> g_job_manager;

// -------------------------
// JSON/字符串工具
// -------------------------
static inline std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '\"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (int)c);
                    o += buf;
                } else {
                    o.push_back((char)c);
                }
        }
    }
    return o;
}

static inline std::string make_tmp_path(const std::string &suffix) {
    std::ostringstream oss;
    oss << "/tmp/ws_ai_upload_" << (long long)time(nullptr)
        << "_" << (long long)rand() << suffix;
    return oss.str();
}

static inline std::string suffix_from_mime(const std::string &mime) {
    std::string m = mime;
    for (auto &c : m) c = (char)tolower((unsigned char)c);
    if (m == "image/png") return ".png";
    if (m == "image/jpeg" || m == "image/jpg") return ".jpg";
    if (m == "image/webp") return ".webp";
    if (m == "image/heic" || m == "image/heif") return ".heic";
    return ".bin";
}

// 极简 JSON 字段提取："key":"value"（仅用于读取 data_url）
static inline std::optional<std::string> json_get_string_field(const std::string &body, const std::string &key) {
    std::string pat = "\"" + key + "\"";
    size_t p = body.find(pat);
    if (p == std::string::npos) return std::nullopt;
    size_t c = body.find(':', p + pat.size());
    if (c == std::string::npos) return std::nullopt;
    size_t q1 = body.find('"', c + 1);
    if (q1 == std::string::npos) return std::nullopt;
    size_t q2 = body.find('"', q1 + 1);
    if (q2 == std::string::npos) return std::nullopt;
    return body.substr(q1 + 1, q2 - (q1 + 1));
}

// data:image/png;base64,xxxx
static inline std::optional<std::pair<std::string, std::string>> parse_data_url(const std::string &data_url) {
    auto p = data_url.find("base64,");
    if (p == std::string::npos) return std::nullopt;
    std::string meta = data_url.substr(0, p);
    std::string payload = data_url.substr(p + 7);

    std::string mime = "application/octet-stream";
    if (meta.rfind("data:", 0) == 0) {
        auto semi = meta.find(';');
        if (semi != std::string::npos) mime = meta.substr(5, semi - 5);
    }
    return std::make_pair(mime, payload);
}

static inline int b64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A');
    if (c >= 'a' && c <= 'z') return (int)(c - 'a') + 26;
    if (c >= '0' && c <= '9') return (int)(c - '0') + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static inline std::optional<std::string> base64_decode(const std::string &in) {
    std::string out;
    out.reserve(in.size() * 3 / 4);

    int val = 0;
    int valb = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        int v = b64_value(c);
        if (v < 0) continue;
        val = (val << 6) + v;
        valb += 6;
        if (valb >= 0) {
            out.push_back((char)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// -------------------------
// 前端页面（你也可以以后改成读静态文件）
// -------------------------
static const char *kIndexHtml = R"HTML(
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ws_ai_tool</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial;margin:18px;}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
.card{border:1px solid #ddd;border-radius:10px;padding:12px;margin-top:12px;}
.btn{padding:8px 12px;border-radius:8px;border:1px solid #333;background:#fff;cursor:pointer;}
.btn:disabled{opacity:.5;cursor:not-allowed;}
.bar{width:100%;height:14px;background:#eee;border-radius:999px;overflow:hidden;}
.bar>div{height:100%;width:0%;background:#111;transition:width .2s linear;}
textarea{width:100%;min-height:260px;}
#drop{border:2px dashed #bbb;border-radius:10px;padding:14px;color:#666;}
img{max-width:360px;border-radius:10px;border:1px solid #ddd;}
code{background:#f6f6f6;padding:2px 6px;border-radius:6px;}
</style>
</head>
<body>
<h3>ws_ai_tool 图片总结</h3>

<div class="card">
  <div class="row">
    <input id="file" type="file" accept="image/*"/>
    <button id="btnUpload" class="btn">上传并开始</button>
    <button id="btnClear" class="btn">清空</button>
  </div>
  <p style="margin:10px 0 0 0;color:#666;">可 Ctrl+V 粘贴或拖拽图片。</p>
  <div id="drop" class="card" style="margin-top:10px;">
    粘贴 / 拖拽图片到这里
    <div style="margin-top:10px;"><img id="preview" style="display:none;"/></div>
  </div>
</div>

<div class="card">
  <div class="row">
    <div style="flex:1;"><div class="bar"><div id="barFill"></div></div></div>
    <div style="min-width:70px;text-align:right;"><span id="pct">0%</span></div>
  </div>
  <p style="margin:8px 0 0 0;color:#666;">状态：<span id="state">idle</span>　任务：<code id="tid">-</code></p>
</div>

<div class="card">
  <h4 style="margin:0 0 8px 0;">输出</h4>
  <textarea id="out" placeholder="这里显示结果..."></textarea>
</div>

<script>
let currentTaskId=null,pastedDataUrl=null,pollingTimer=null;
function setProgress(p){p=Math.max(0,Math.min(100,p|0));barFill.style.width=p+'%';pct.textContent=p+'%';}
function setState(s){state.textContent=s;}
function setTaskId(id){tid.textContent=id||'-';}
function stopPolling(){if(pollingTimer){clearInterval(pollingTimer);pollingTimer=null;}}
function startPolling(taskId){
  stopPolling();
  pollingTimer=setInterval(async()=>{
    try{
      const r=await fetch('/api/status?id='+encodeURIComponent(taskId));
      const j=await r.json();
      setProgress(j.progress||0);
      setState(j.state||'unknown');
      if(j.state==='done'){stopPolling();out.value=j.result||'';}
      else if(j.state==='error'){stopPolling();out.value=j.error||'error';}
    }catch(e){}
  },250);
}
function showPreview(dataUrl){preview.src=dataUrl;preview.style.display='inline-block';}
async function uploadFileAndStart(file){
  const fd=new FormData(); fd.append('file',file);
  setState('uploading'); setProgress(1);
  const r=await fetch('/api/upload',{method:'POST',body:fd});
  const j=await r.json(); if(!j.ok) throw new Error(j.error||'upload failed');
  currentTaskId=j.id; setTaskId(currentTaskId); setState('queued'); setProgress(3); startPolling(currentTaskId);
}
async function uploadDataUrlAndStart(dataUrl){
  setState('uploading'); setProgress(1);
  const r=await fetch('/api/clipboard',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data_url:dataUrl})});
  const j=await r.json(); if(!j.ok) throw new Error(j.error||'clipboard upload failed');
  currentTaskId=j.id; setTaskId(currentTaskId); setState('queued'); setProgress(3); startPolling(currentTaskId);
}
btnUpload.onclick=async()=>{
  try{
    out.value='';
    if(pastedDataUrl){await uploadDataUrlAndStart(pastedDataUrl);return;}
    const f=file.files[0]; if(!f){alert('请选择文件或粘贴图片');return;}
    await uploadFileAndStart(f);
  }catch(e){setState('error');out.value=String(e);}
};
btnClear.onclick=()=>{
  pastedDataUrl=null;currentTaskId=null;stopPolling();
  setProgress(0);setState('idle');setTaskId(null);out.value='';
  preview.style.display='none';preview.src='';file.value='';
};
window.addEventListener('paste',(ev)=>{
  try{
    const items=(ev.clipboardData||ev.originalEvent.clipboardData).items;
    for(const it of items){
      if(it.type&&it.type.startsWith('image/')){
        const blob=it.getAsFile(); const reader=new FileReader();
        reader.onload=()=>{pastedDataUrl=reader.result;showPreview(pastedDataUrl);};
        reader.readAsDataURL(blob); ev.preventDefault(); return;
      }
    }
  }catch(e){}
});
drop.addEventListener('dragover',(e)=>{e.preventDefault();drop.style.borderColor='#111';});
drop.addEventListener('dragleave',()=>{drop.style.borderColor='#bbb';});
drop.addEventListener('drop',(e)=>{
  e.preventDefault();drop.style.borderColor='#bbb';
  const f=e.dataTransfer.files&&e.dataTransfer.files[0];
  if(!f) return; if(!f.type.startsWith('image/')){alert('请拖拽图片文件');return;}
  const reader=new FileReader();
  reader.onload=()=>{pastedDataUrl=reader.result;showPreview(pastedDataUrl);};
  reader.readAsDataURL(f);
});
</script>
</body>
</html>
)HTML";

// -------------------------
// 关键：补齐你链接缺的两个符号（必须与你头文件签名一致）
// -------------------------
HttpServer::HttpServer(Config cfg, std::shared_ptr<JobManager> jm)
: cfg_(std::move(cfg)) {
    g_job_manager = std::move(jm);
    std::srand((unsigned)time(nullptr));
}

static inline bool ensure_job_manager(httplib::Response &res) {
    if (!g_job_manager) {
        res.status = 500;
        res.set_content("{\"ok\":false,\"error\":\"JobManager is null\"}", "application/json; charset=utf-8");
        return false;
    }
    return true;
}

// -------------------------
// serve_forever：启动 8080 服务
// -------------------------
void HttpServer::serve_forever() {
    httplib::Server svr;

    // 首页
    svr.Get("/", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(kIndexHtml, "text/html; charset=utf-8");
    });

    // 查询状态：GET /api/status?id=xxx
    svr.Get("/api/status", [&](const httplib::Request &req, httplib::Response &res) {
        if (!ensure_job_manager(res)) return;
        if (!req.has_param("id")) {
            res.status = 400;
            res.set_content("{\"ok\":false,\"error\":\"missing id\"}", "application/json; charset=utf-8");
            return;
        }
        const std::string id = req.get_param_value("id");

        // 你需要在 JobManager 实现这个函数（或改成你已有的接口）
        const std::string json = g_job_manager->get_status_json(id);
        res.set_content(json, "application/json; charset=utf-8");
    });

    // 上传文件：POST /api/upload  multipart/form-data name="file"
    // 注意：你当前的 httplib 版本不支持 req.files/has_file/get_file_value，所以必须用 ContentReader 解析
    svr.Post("/api/upload",
        [&](const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &content_reader) {
            (void)req;
            if (!ensure_job_manager(res)) return;

            bool got_file = false;
            std::string filename;
            std::string content_type;
            std::string file_bytes;

            bool ok = content_reader(
                [&](const httplib::FormData &header) {
                    if (header.name == "file") {
                        got_file = true;
                        filename = header.filename;
                        content_type = header.content_type;
                        file_bytes.clear();
                    }
                    return true;
                },
                [&](const char *data, size_t data_length) {
                    if (got_file) file_bytes.append(data, data_length);
                    return true;
                }
            );

            if (!ok || !got_file || file_bytes.empty()) {
                res.status = 400;
                res.set_content("{\"ok\":false,\"error\":\"missing file\"}", "application/json; charset=utf-8");
                return;
            }

            std::string suffix = ".bin";
            auto dot = filename.find_last_of('.');
            if (!filename.empty() && dot != std::string::npos) suffix = filename.substr(dot);
            else if (!content_type.empty()) suffix = suffix_from_mime(content_type);

            const std::string save_path = make_tmp_path(suffix);
            {
                std::ofstream ofs(save_path, std::ios::binary);
                if (!ofs) {
                    res.status = 500;
                    res.set_content("{\"ok\":false,\"error\":\"failed to write temp file\"}", "application/json; charset=utf-8");
                    return;
                }
                ofs.write(file_bytes.data(), (std::streamsize)file_bytes.size());
            }

            // 你需要在 JobManager 实现这个函数（或改成你已有的接口）
            const std::string id = g_job_manager->submit_image(save_path);

            std::ostringstream oss;
            oss << "{\"ok\":true,\"id\":\"" << json_escape(id) << "\"}";
            res.set_content(oss.str(), "application/json; charset=utf-8");
        }
    );

    // 剪贴板 dataURL：POST /api/clipboard  JSON {"data_url":"data:image/png;base64,..."}
    svr.Post("/api/clipboard", [&](const httplib::Request &req, httplib::Response &res) {
        if (!ensure_job_manager(res)) return;

        if (req.body.empty()) {
            res.status = 400;
            res.set_content("{\"ok\":false,\"error\":\"empty body\"}", "application/json; charset=utf-8");
            return;
        }

        auto data_url_opt = json_get_string_field(req.body, "data_url");
        if (!data_url_opt) {
            res.status = 400;
            res.set_content("{\"ok\":false,\"error\":\"missing data_url\"}", "application/json; charset=utf-8");
            return;
        }

        auto parsed = parse_data_url(*data_url_opt);
        if (!parsed) {
            res.status = 400;
            res.set_content("{\"ok\":false,\"error\":\"invalid data_url\"}", "application/json; charset=utf-8");
            return;
        }

        const std::string mime = parsed->first;
        const std::string b64  = parsed->second;

        auto bin_opt = base64_decode(b64);
        if (!bin_opt) {
            res.status = 400;
            res.set_content("{\"ok\":false,\"error\":\"base64 decode failed\"}", "application/json; charset=utf-8");
            return;
        }

        const std::string save_path = make_tmp_path(suffix_from_mime(mime));
        {
            std::ofstream ofs(save_path, std::ios::binary);
            if (!ofs) {
                res.status = 500;
                res.set_content("{\"ok\":false,\"error\":\"failed to write temp file\"}", "application/json; charset=utf-8");
                return;
            }
            ofs.write(bin_opt->data(), (std::streamsize)bin_opt->size());
        }

        const std::string id = g_job_manager->submit_image(save_path);

        std::ostringstream oss;
        oss << "{\"ok\":true,\"id\":\"" << json_escape(id) << "\"}";
        res.set_content(oss.str(), "application/json; charset=utf-8");
    });

    // 监听地址：默认 0.0.0.0:8080
    // 如果你想改端口：export WS_AI_PORT=8090
    // 如果你想改 host： export WS_AI_HOST=127.0.0.1
    std::string host = "0.0.0.0";
    int port = 8080;

    if (const char *h = std::getenv("WS_AI_HOST")) {
        if (std::string(h).size() > 0) host = h;
    }
    if (const char *p = std::getenv("WS_AI_PORT")) {
        int v = std::atoi(p);
        if (v > 0 && v < 65536) port = v;
    }

    std::cout << "Listening on http://" << host << ":" << port << "\n";
    svr.listen(host.c_str(), port);
}

} // namespace ws_ai