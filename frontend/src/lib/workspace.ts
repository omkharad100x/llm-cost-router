import {useCallback,useEffect,useRef,useState} from "react";
export type RequestRow={time:string;model:string;kind:string;status:string;ms:number|null;cost:string|null;input:number|null;output:number|null};
type Count = number | null;
type Roles = {small: Count; large: Count};
type Cache = {
  t: number; pairs: number; hits: Count; misses: Count; hitRate: Count; missRate: Count;
  falseHits: Count; falseMisses: Count; falseHitRate: Count; falseMissRate: Count;
  seeded: Count; lookups: Count; avoided: Count; lookup: Count;
};
type MoneyDisplay = {estimated_cost_usd: string|null; known_cost_subtotal_usd: string|null; average_cost_per_query_usd: string|null};
type ComponentCost = {estimated_cost_usd: string|null; known_cost_subtotal_usd:string; unknown_cost_requests:number; display:MoneyDisplay};
type CostSummary = {
  display: MoneyDisplay;
  components: Record<"input_cost_usd"|"cached_input_cost_usd"|"output_cost_usd", ComponentCost>;
  estimated_cost_usd: string | null; known_cost_subtotal_usd: string;
  average_cost_per_query_usd: string | null; unknown_cost_requests: number;
  complete: boolean; requests: number; generation_calls: number;
};
type CostComparison = {status: "missing" | "invalid"; issues: string[]} | {
  status: "ready"; issues: string[]; query_count: number; threshold: number;
  baseline: CostSummary; optimized: CostSummary; escalation: CostSummary; embedding: CostSummary;
  savings_display:string|null; savings_percentage_display:string|null;
  savings_usd: string | null; savings_percentage: string | null; safe_calls_avoided: number;
  cache_outcomes: {safe_hit: number; false_hit: number; unreviewed_hit: number; miss: number};
  by_model: {model: string; baseline: CostSummary; optimized: CostSummary}[];
};
type Benchmark = {
  comparison: CostComparison;
  performance: any;
  cache: Cache[]; queries: number; small: Count; large: Count; simple: Count; complex: Count;
  attempts: Roles; final: Roles; escalations: Count; successes: Count; accepted: Count; passed: Count;
  escalationRate: Count; successRate: Count; acceptanceRate: Count;
};
type HistoricalCosts = CostSummary & {
  by_model: (CostSummary & {model:string})[]; embedding:CostSummary;
  daily:{date:string;cost:string|null;display:string|null}[];
};
type Workspace={history:RequestRow[]|null;costs:HistoricalCosts|null;hs:string;bench:Benchmark|null;bs:string;small:string;large:string;schema:number;version:string;threshold:number;evaluation:number};
export function useWorkspace(options:Record<string,string>){
 const [data,setData]=useState<Workspace|null>(null),[loading,setLoading]=useState(true),[error,setError]=useState("");
 const active = useRef<AbortController|null>(null);
 const query = new URLSearchParams(options).toString();
 const refresh = useCallback(async()=>{
   active.current?.abort();
   const controller = new AbortController();
   active.current = controller;
   setLoading(true);
   try {
     const response = await fetch("/api/workspace?"+query,{cache:"no-store",signal:controller.signal});
     if(!response.ok) throw Error();
     const next = await response.json();
     if(!controller.signal.aborted){setData(next);setError("");}
   } catch {
     if(!controller.signal.aborted){setData(null);setError("Workspace unavailable. Refresh to retry.");}
   } finally {
     if(!controller.signal.aborted) setLoading(false);
   }
 },[query]);
 useEffect(()=>{void refresh();return ()=>active.current?.abort()},[refresh]);
 return {data,loading,error,refresh};
}
