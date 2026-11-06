import {defineConfig,type Plugin} from "vite";
import react from "@vitejs/plugin-react";
import {execFile} from "node:child_process";
import {fileURLToPath} from "node:url";
const local=(path:string)=>fileURLToPath(new URL(path,import.meta.url));
const api: NonNullable<Plugin["configureServer"]> = server=>{server.middlewares.use("/api/workspace",(req,res)=>{const query=new URL(req.url??"/","http://127.0.0.1").searchParams;const options=Object.fromEntries(["from","to","model","currency","rate"].flatMap(key=>query.has(key)?[[key,query.get(key)!]]:[]));execFile(local(process.platform==="win32"?"../.venv/Scripts/python.exe":"../.venv/bin/python"),["-B",local("./workspace_data.py"),JSON.stringify(options)],{timeout:5000},(err,out)=>{res.setHeader("Content-Type","application/json");res.setHeader("Cache-Control","no-store");res.statusCode=err?503:200;res.end(err?"{}":out)})})};
export default defineConfig({plugins:[react(),{name:"local-workspace",configureServer:api,configurePreviewServer:api}],resolve:{alias:{"@":local("./src")}},server:{host:"127.0.0.1",port:5173,strictPort:true,fs:{strict:true,allow:[local("./src"),local("./node_modules"),local("./index.html")]}}});
