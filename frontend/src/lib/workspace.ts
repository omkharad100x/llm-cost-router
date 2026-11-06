import {useCallback,useEffect,useState} from "react";
export type RequestRow={time:string;model:string;kind:string;status:string;ms:number|null;cost:number|null;input:number|null;output:number|null};
type Count = number | null;
type Roles = {small: Count; large: Count};
type Cache = {
  t: number; pairs: number; hits: Count; misses: Count; hitRate: Count; missRate: Count;
  falseHits: Count; falseMisses: Count; falseHitRate: Count; falseMissRate: Count;
  seeded: Count; lookups: Count; avoided: Count; lookup: Count;
};
type CostSummary = {
  estimated_cost_usd: string | null; known_cost_subtotal_usd: string;
  average_cost_per_query_usd: string | null; unknown_cost_requests: number;
  complete: boolean; requests: number; generation_calls: number;
};
type CostComparison = {status: "missing" | "invalid"; issues: string[]} | {
  status: "ready"; issues: string[]; query_count: number; threshold: number;
  baseline: CostSummary; optimized: CostSummary; escalation: CostSummary; embedding: CostSummary;
  savings_usd: string | null; savings_percentage: string | null; safe_calls_avoided: number;
  cache_outcomes: {safe_hit: number; false_hit: number; unreviewed_hit: number; miss: number};
  by_model: {model: string; baseline: CostSummary; optimized: CostSummary}[];
};
type Benchmark = {
  comparison: CostComparison;
  cache: Cache[]; queries: number; small: Count; large: Count; simple: Count; complex: Count;
  attempts: Roles; final: Roles; escalations: Count; successes: Count; accepted: Count; passed: Count;
  escalationRate: Count; successRate: Count; acceptanceRate: Count;
};
type Workspace={history:RequestRow[]|null;hs:string;bench:Benchmark|null;bs:string;small:string;large:string;schema:number;version:string;threshold:number;evaluation:number};
export function useWorkspace(){
 const [data,setData]=useState<Workspace|null>(null),[loading,setLoading]=useState(true),[error,setError]=useState("");
 const refresh=useCallback(async()=>{setLoading(true);try{const r=await fetch("/api/workspace",{cache:"no-store"});if(!r.ok)throw Error();setData(await r.json());setError("")}catch{setData(null);setError("Workspace unavailable. Refresh to retry.")}finally{setLoading(false)}},[]);
 useEffect(()=>{void refresh()},[refresh]);return {data,loading,error,refresh};
}