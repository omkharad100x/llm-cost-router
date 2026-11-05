import {useCallback,useEffect,useState} from "react";
export type RequestRow={time:string;model:string;kind:string;status:string;ms:number|null;cost:number|null;input:number|null;output:number|null};
type Cache={t:number;hits:number;misses:number;falseHits:number;falseMisses:number;avoided:number;lookup:number};
type Benchmark={cache:Cache[];small:number;large:number;final:{small:number;large:number};simple:number;complex:number;queries:number;escalations:number;successes:number;accepted:number;passed:number};
type Workspace={history:RequestRow[]|null;hs:string;bench:Benchmark|null;bs:string;small:string;large:string;schema:number;version:string;threshold:number;evaluation:number};
export function useWorkspace(){
 const [data,setData]=useState<Workspace|null>(null),[loading,setLoading]=useState(true),[error,setError]=useState("");
 const refresh=useCallback(async()=>{setLoading(true);try{const r=await fetch("/api/workspace",{cache:"no-store"});if(!r.ok)throw Error();setData(await r.json());setError("")}catch{setData(null);setError("Workspace unavailable. Refresh to retry.")}finally{setLoading(false)}},[]);
 useEffect(()=>{void refresh()},[refresh]);return {data,loading,error,refresh};
}