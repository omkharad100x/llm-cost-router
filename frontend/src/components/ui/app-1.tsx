import {useState,useSyncExternalStore,type ReactNode} from "react";
import {Search,SlidersHorizontal,ArrowUpRight,Sparkles,DatabaseZap,Activity,Database,RefreshCw,Route,Wallet,Settings,LayoutDashboard} from "lucide-react";
import {BarChart,Bar,AreaChart,Area,LineChart,Line,CartesianGrid,XAxis,YAxis,Tooltip,ResponsiveContainer} from "recharts";
import {useWorkspace,type RequestRow} from "@/lib/workspace";
import {Button} from "./button";
const pages=["Overview","Cache Analytics","Model Routing","Cost Analysis","Performance","System Information"];
const icons=[LayoutDashboard,Database,Route,Wallet,Activity,Settings];
const day=(d:Date)=>d.toISOString().slice(0,10);
const percent=(fraction:number|null|undefined)=>fraction==null?null:(100*fraction).toFixed(1)+"%";
const ratio=(n:number|null|undefined,d:number|null|undefined)=>n!=null&&d? (100*n/d).toFixed(1)+"%":null;
const sum=(v:(number|null)[])=>v.length&&v.every((x):x is number=>x!=null)?v.reduce((a,b)=>a+b,0):null;
const avg=(v:number[])=>v.length?v.reduce((a,b)=>a+b,0)/v.length:null;
const ms=(v:number|null|undefined)=>v==null?null:v.toFixed(1)+" ms";
const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
function subscribeToMotion(change: () => void) {
  motionPreference.addEventListener("change", change);
  return () => motionPreference.removeEventListener("change", change);
}
const chartTick = { fontSize: 12, fill: "#53685e" };
const tooltipStyle = { borderRadius: 12, border: "1px solid #d5dfd8", fontSize: 13 };

function Tile({label, value, source}: {label:string; value:ReactNode; source:string}) {
  return (
    <article className="tile" data-available={value != null}>
      <h3>{label}</h3>
      <strong key={String(value)} className="metric-value">{value ?? "Unavailable"}</strong>
      <small className="source-label" data-source={source}>{source === "Benchmark" ? "Synthetic benchmark" : source}</small>
    </article>
  );
}

function Panel({title, note, children}: {title:string; note:string; children:ReactNode}) {
  return <section className="panel"><h2>{title}</h2><small>{note}</small>{children}</section>;
}

function Split({labels, values, missing = "Report counts unavailable."}: {labels:string[]; values:(number|null|undefined)[]; missing?:string}) {
  if (values.every(value => value == null)) return <p className="empty">{missing}</p>;
  const total = sum(values.map(value => value ?? null));
  return <dl className="bars">{labels.map((label, i) => (
    <div key={label}>
      <div className="line"><dt>{label}</dt><dd>{values[i] ?? "Unavailable"} {total != null && total > 0 && <small>({ratio(values[i], total)})</small>}</dd></div>
      {total != null && total > 0 && <progress aria-label={label} max={total} value={values[i] ?? undefined}/>}
    </div>
  ))}</dl>;
}
export default function App1(){
 const reducedMotion = useSyncExternalStore(subscribeToMotion, () => motionPreference.matches);
 const [page,P]=useState(0),[range,R]=useState("7 days"),[model,M]=useState("All models"),[currency,C]=useState("USD"),[rate,F]=useState("1"),[from,A]=useState(day(new Date())),[to,Z]=useState(day(new Date())),[threshold,T]=useState<number|null>(null);
 const today=day(new Date()),start=range==="Custom"?from:day(new Date(Date.now()-86400000*(range==="Today"?0:range==="7 days"?6:29))),end=range==="Custom"?to:today;
 const {data,loading,error,refresh}=useWorkspace({from:start,to:end,model,currency,rate:currency==="USD"?"1":rate});
 const costs=loading?null:data?.costs;
 const invalid=!start||!end||start>end,rows=(data?.history??[]).filter(r=>!invalid&&r.time.slice(0,10)>=start&&r.time.slice(0,10)<=end&&(model==="All models"||r.model===model)).sort((a,b)=>b.time.localeCompare(a.time));
 const b=data?.bench,c=b?.cache.find(x=>x.t===(threshold??data?.threshold))??b?.cache[0],n=data?.history&&!invalid?rows.length:null;
 const comparison=b?.comparison.status==="ready"?b.comparison:null;
 const factor=currency==="USD"?1:Number(rate);
 const lat=rows.flatMap(r=>r.ms==null?[]:[r.ms]).sort((a,b)=>a-b),pct=(p:number)=>lat.length?lat[Math.ceil(lat.length*p)-1]:null;
 const models=[...new Set([data?.small,data?.large,...(data?.history??[]).map(r=>r.model)].filter((x):x is string=>!!x))];
 const daily=new Map<string,RequestRow[]>();for(const r of rows){const d=r.time.slice(0,10);daily.set(d,[...(daily.get(d)??[]),r])}
 const trend=[...daily].sort(([a],[b])=>a.localeCompare(b)).map(([date,rs])=>({date,requests:rs.length}));
 const tile=(l:string,v:ReactNode,s="Historical")=><Tile key={l} label={l} value={v} source={s}/>;
 const bench=(l:string,v:ReactNode)=>tile(l,v,"Benchmark"); const unavailable=(l:string)=>tile(l,null,"Unmeasured");
 const convertedTrend = trend.map(point => {
   const dailyCost = costs?.daily.find(item=>item.date===point.date);
   return {...point,cost:dailyCost?.cost==null?null:Number(dailyCost.cost),costDisplay:dailyCost?.display};
 });
 const chart = (cost = false) => {
   const hasPoints = cost ? convertedTrend.some(point => point.cost != null) : trend.length > 0;
   return (
     <Panel title={cost ? "Cost over time" : "Request activity"}
            note={cost ? "Historical · estimated daily cost in " + currency : "Historical · API requests per UTC day"}>
       {hasPoints ? <ResponsiveContainer width="100%" height={196}>
         <AreaChart accessibilityLayer data={convertedTrend} margin={{top:12, right:14, bottom:0, left:2}}>
           <CartesianGrid vertical={false} stroke="#dce5df" strokeDasharray="3 4"/>
           <XAxis dataKey="date" axisLine={false} tickLine={false} tick={chartTick}
                  minTickGap={24} tickMargin={8} tickFormatter={date => String(date).slice(5)}/>
           <YAxis width={56} axisLine={false} tickLine={false} tick={chartTick} allowDecimals={cost}/>
           <Tooltip contentStyle={tooltipStyle} formatter={(value,_name,item)=>cost?(item.payload.costDisplay??"Unavailable"):value}/>
           <Area name={cost ? "Estimated cost (" + currency + ")" : "Requests"} dataKey={cost ? "cost" : "requests"}
                 stroke="#215740" strokeWidth={2} fill="#abdc8f" fillOpacity={0.32} dot={{r:3}}
                 isAnimationActive={!reducedMotion} animationDuration={220}/>
         </AreaChart>
       </ResponsiveContainer> : <p className="empty">{cost && trend.length ? "No complete daily cost estimates for this selection." : historyEmpty}</p>}
     </Panel>
   );
 };
 const [filters,showFilters]=useState(false),[search,setSearch]=useState("");
 const [searchMessage,setSearchMessage]=useState("");
 const historicScope=(range==="Custom"?from+" – "+to:range)+" · "+model+" · "+currency;
 const historyEmpty=invalid?"Choose a valid custom date range.":loading?"Reading local records…":data?.hs==="Missing"?"Request history will appear after the first local request log is available.":data?.hs!=="Ready"?"Request history is unavailable. Check the local source and refresh.":"No requests match the selected dates and model.";
 const findPage=(e:React.FormEvent)=>{e.preventDefault();if(!search.trim()){setSearchMessage("Enter a section name.");return}const hit=pages.findIndex(s=>s.toLowerCase().includes(search.trim().toLowerCase()));if(hit>=0){P(hit);setSearch("");setSearchMessage("")}else{setSearchMessage("No matching section.")}};
 return <main className="shell"><header className="mast"><div className="brand"><Route/> Cost Router</div>
 <nav aria-label="Dashboard sections">{pages.map((s,i)=><button aria-current={i===page?"page":undefined} key={s} onClick={()=>P(i)}>{s}</button>)}</nav>
 <form className="search" onSubmit={findPage}><button aria-label="Search sections"><Search size={16}/></button><input aria-label="Find a section" list="sections" value={search} onChange={e=>{setSearch(e.target.value);setSearchMessage("")}} placeholder="Search"/><datalist id="sections">{pages.map(x=><option key={x}>{x}</option>)}</datalist>{searchMessage&&<output className="search-result">{searchMessage}</output>}</form>
 <button title="Dashboard filters" aria-expanded={filters} onClick={()=>showFilters(!filters)}><SlidersHorizontal size={20}/></button></header>
 <div className="desk"><aside className="rail"><div><button title="Refresh data" disabled={loading} onClick={refresh}><RefreshCw/></button><button title="Filters" onClick={()=>showFilters(!filters)}><SlidersHorizontal/></button></div><div>{icons.map((Ic,i)=><button title={pages[i]} key={i} aria-current={page===i?"page":undefined} onClick={()=>P(i)}><Ic/></button>)}</div></aside><div className="workspace" data-loading={loading}>
 {page!==0&&<div className="section-title"><h1>{pages[page]}</h1><button onClick={()=>showFilters(!filters)} aria-expanded={filters}><SlidersHorizontal size={15}/><span key={historicScope} className="scope-label">{historicScope}</span></button></div>}
 {filters&&<section className="filterbox"><div className="filter-heading"><h2>Dashboard filters</h2><button className="reset" onClick={()=>{R("7 days");M("All models");C("USD");F("1");A(today);Z(today)}}>Reset filters</button><Button disabled={loading} onClick={refresh}><RefreshCw size={15}/> {loading ? "Refreshing…" : "Refresh data"}</Button></div>
 <div className="controls"><label>Date range<select value={range} onChange={e=>R(e.target.value)}>{["Today","7 days","30 days","Custom"].map(x=><option key={x}>{x}</option>)}</select></label>{range==="Custom"&&<><label>From<input type="date" value={from} onChange={e=>A(e.target.value)}/></label><label>To<input type="date" value={to} onChange={e=>Z(e.target.value)}/></label></>}
 <label>Model<select value={model} onChange={e=>M(e.target.value)}>{["All models",...models].map(x=><option key={x}>{x}</option>)}</select></label>
 <label>Currency<select value={currency} onChange={e=>{C(e.target.value);F("")}}>{["USD","EUR","INR","GBP"].map(x=><option key={x}>{x}</option>)}</select></label>{currency!=="USD"&&<label>{currency} per USD<input type="number" min="0" step="any" value={rate} onChange={e=>F(e.target.value)} placeholder="Enter rate"/></label>}</div>
 </section>}
 {error&&<p role="alert">{error}</p>}
 <div className="page-content" key={page}>
 {page===0&&<div className="fynix">
 <section className="welcome surface"><div className="greet"><span><Route size={18}/></span><h1>Workspace overview</h1></div>
 <div className="hero-number">{b?.queries??"—"}</div><p className="center">Routing queries · synthetic benchmark</p>
 <div className="hero-actions"><button onClick={()=>P(2)}>Routing <ArrowUpRight size={16}/></button><button onClick={()=>P(1)}>Cache <Database size={16}/></button></div>
 <div className="wallet-grid"><article><Route/><strong>{b?.simple??"—"}</strong>Simple queries</article><article><LayoutDashboard/><strong>{b?.complex??"—"}</strong>Complex queries</article><article className="lime"><DatabaseZap/><strong>{c?.avoided??"—"}</strong>API calls avoided</article><article><Activity/><strong>{b?.escalations??"—"}</strong>Escalations</article></div></section>
 <div className="four surface">{tile("Total requests",n)}{tile("Estimated API cost",costs?.display.estimated_cost_usd,"Historical estimate")}{bench("Cache hit rate",percent(c?.hitRate))}{unavailable("Estimated cost avoided")}</div>
 <section className="guide surface"><h2>Router workspace</h2><div className="orb" aria-hidden="true"/><strong>Explore your routing data</strong><div className="suggest">{[1,2,3].map(i=><button key={i} onClick={()=>P(i)}>{pages[i]} ↗</button>)}</div><div className="guide-bar"><Sparkles size={17}/><span key={historicScope}>{historicScope}</span><button onClick={()=>showFilters(!filters)}>Filters <SlidersHorizontal size={15}/></button></div></section>
 <section className="mix surface"><h2>Model distribution</h2><small>Synthetic benchmark · initial selections</small><div className="mix-body"><dl className="legend"><div><dt>Small</dt><dd>{b?.small??"—"}</dd></div><div><dt>Large</dt><dd>{b?.large??"—"}</dd></div></dl><div className="donut" style={{background:b?.queries && b.small!=null && b.large!=null?'conic-gradient(#09362e 0 '+(100*b.small/b.queries)+'%, #a5e773 0)':'#e8ebe7'}}><span>{b?.queries??"—"}<small>queries</small></span></div></div></section>
 <section className="balance surface"><h2>Cache outcomes</h2><small>Synthetic benchmark · threshold {c?.t??"—"}</small><div className="outcomes"><div><Database/> Hits <b>{c?.hits??"—"}</b></div><div><Route/> Misses <b>{c?.misses??"—"}</b></div></div></section>
 <section className="activity surface"><h2>Recent system activity</h2><small>Historical · newest first</small>{rows.length>0&&<table><thead><tr><th>Request</th><th>UTC time</th><th>Status</th></tr></thead><tbody>{rows.slice(0,5).map((r,i)=><tr key={i}><td>{r.kind}</td><td>{r.time.slice(0,19)}</td><td>{r.status}</td></tr>)}</tbody></table>}{!rows.length&&<p className="empty">{historyEmpty}</p>}</section>
 <div className="flow">{chart()}</div></div>}
 {page===1&&<>
 <label className="pick">Benchmark similarity threshold
   <select value={c?.t??""} disabled={!b?.cache.length} onChange={e=>T(Number(e.target.value))}>
     {!b?.cache.length && <option value="">Unavailable</option>}
     {b?.cache.map(x=><option key={x.t}>{x.t}</option>)}
   </select>
 </label>
 <p className="context">Synthetic benchmark · {c ? c.pairs+" reviewed pairs, one outcome per pair." : "Reviewed pair counts unavailable."}</p>
 <div className="tiles">
   {bench("Cache hits",c?.hits)}{bench("Cache misses",c?.misses)}
   {bench("Hit rate",percent(c?.hitRate))}{bench("Miss rate",percent(c?.missRate))}
 </div>
 <div className="tiles">
   {bench("Cached fixture responses",c?.seeded)}{bench("Generation calls avoided",c?.avoided)}
   {bench("Total lookups",c?.lookups)}{bench("Average cache lookup",ms(c?.lookup))}
 </div>
 <p className="context">Avoided calls include repeated lookups and false hits. Lookup latency covers SQLite, mock embedding and search.</p>
 <div className="panels threshold-panels">
 <Panel title="False matches by threshold" note="Synthetic benchmark · reviewed pairs">
 {b?.cache.some(x => x.falseHits != null || x.falseMisses != null) ? <>
   <div className="series-key"><span>False hits</span><span>False misses</span></div>
   <ResponsiveContainer width="100%" height={176}>
     <LineChart accessibilityLayer data={b.cache} margin={{top:10,right:22,left:0,bottom:4}}>
       <CartesianGrid vertical={false} stroke="#dce5df" strokeDasharray="3 4"/>
       <XAxis dataKey="t" axisLine={false} tickLine={false} tick={chartTick} tickMargin={8}/>
       <YAxis allowDecimals={false} width={36} axisLine={false} tickLine={false} tick={chartTick}/>
       <Tooltip contentStyle={tooltipStyle} labelFormatter={t=>"Threshold " + t}/>
       <Line dataKey="falseHits" name="False hits" stroke="#174f3d" strokeWidth={2} dot={{r:3}} isAnimationActive={!reducedMotion} animationDuration={220}/>
       <Line dataKey="falseMisses" name="False misses" stroke="#737c2e" strokeDasharray="5 3" strokeWidth={2} dot={{r:3}} isAnimationActive={!reducedMotion} animationDuration={220}/>
     </LineChart>
   </ResponsiveContainer>
 </> : <p className="empty">The report has no false-hit or false-miss counts.</p>}</Panel>
 <Panel title="Similarity threshold analysis" note="Synthetic benchmark · reviewed query pairs"><table><thead><tr>{["Threshold","Hits","Misses","False hits","False misses"].map(x=><th key={x}>{x}</th>)}</tr></thead><tbody>{b?.cache.map(x=><tr key={x.t} data-selected={x.t===c?.t} aria-current={x.t===c?.t ? "true" : undefined}>{[x.t,x.hits,x.misses,x.falseHits,x.falseMisses].map((v,i)=><td key={i}>{v ?? "—"}</td>)}</tr>)}</tbody></table>{!b&&<p>Benchmark unavailable.</p>}</Panel></div>
 <p className="context">False-hit rate among must-not-match pairs <strong>{percent(c?.falseHitRate) ?? "Unavailable"}</strong>
   {" · "}False-miss rate among safe-match pairs <strong>{percent(c?.falseMissRate) ?? "Unavailable"}</strong></p>
 <div className="tiles">{unavailable("Current cached responses")}{unavailable("Cache storage usage")}</div>
 <p className="context">Historical cache metrics unavailable. The request log omits cache decisions, lookup times and avoided calls. Neither source measures current cached responses or cache storage usage. A dash indicates a missing count.</p>
 </>}
 {page===2&&<>
 <div className="tiles">
   {bench("Escalations",b?.escalations)}{bench("Escalation rate",percent(b?.escalationRate))}
   {bench("Initial answer acceptance",percent(b?.acceptanceRate))}{bench("Escalation success",percent(b?.successRate))}
 </div>
 <p className="context">Escalation and initial acceptance rates use all benchmark queries. Escalation success counts passing second answers per triggered escalation.</p>
 <div className="panels">
   <Panel title="Query complexity" note="Synthetic benchmark · classified queries"><Split labels={["Simple","Complex"]} values={[b?.simple,b?.complex]}/></Panel>
   <Panel title="Initial model selection frequency" note="Synthetic benchmark · first role per query"><Split labels={["Small","Large"]} values={[b?.small,b?.large]}/></Panel>
   <Panel title="All model attempts" note="Synthetic benchmark · includes escalation calls"><Split labels={["Small","Large"]} values={[b?.attempts.small,b?.attempts.large]}/></Panel>
   <Panel title="Final model selection" note="Synthetic benchmark · role of the returned answer"><Split labels={["Small","Large"]} values={[b?.final.small,b?.final.large]}/></Panel>
   <Panel title="Recorded model attempts" note="Historical · role-tagged requests in the selected dates and model">
     <Split labels={["Small","Large"]} missing={historyEmpty} values={["small_model","large_model"].map(kind=>n==null?null:rows.filter(r=>r.kind===kind).length)}/>
   </Panel>
 </div>
 <p className="context">Historical counts include successful and failed role-tagged calls. The request log omits classifications, linked attempts, evaluation outcomes and final selections.</p>
 </>}
 {page===3&&<>
 <p className="context">Synthetic benchmark estimates · USD · {comparison?.query_count ?? "Unavailable"} paired queries · cold cache start</p>
 <div className="tiles">
   {bench("Baseline cost",comparison?.baseline.display.estimated_cost_usd)}
   {bench("Optimized cost",comparison?.optimized.display.estimated_cost_usd)}
   {bench("Estimated savings",comparison?.savings_display)}
   {bench("Savings percentage",comparison?.savings_percentage_display)}
 </div>
 <div className="tiles">
   {bench("Baseline cost per query",comparison?.baseline.display.average_cost_per_query_usd)}
   {bench("Optimized cost per query",comparison?.optimized.display.average_cost_per_query_usd)}
 </div>
 {(b?.comparison.issues ?? ["The paired benchmark report is unavailable."]).map(issue=><p className="empty" key={issue}>{issue}</p>)}
 {comparison&&<>
 <div className="panels">
   <Panel title="Scenario comparison" note="Synthetic benchmark · generation calls, including escalations">
     <ResponsiveContainer width="100%" height={210}>
       <BarChart accessibilityLayer data={[{scenario:"Baseline",calls:comparison.baseline.generation_calls},{scenario:"Optimized",calls:comparison.optimized.generation_calls}]} margin={{top:12,right:16,left:0,bottom:8}}>
         <CartesianGrid vertical={false} stroke="#dce5df" strokeDasharray="3 4"/>
         <XAxis dataKey="scenario" tick={chartTick} axisLine={false} tickLine={false}/>
         <YAxis allowDecimals={false} tick={chartTick} axisLine={false} tickLine={false} width={36}/>
         <Tooltip contentStyle={tooltipStyle}/>
         <Bar name="Generation calls" dataKey="calls" fill="#215740" radius={[5,5,0,0]} isAnimationActive={!reducedMotion} animationDuration={220}/>
       </BarChart>
     </ResponsiveContainer>
   </Panel>
   <Panel title="Estimated scenario cost" note="Synthetic benchmark · complete totals in USD">
     {comparison.baseline.complete&&comparison.optimized.complete?<ResponsiveContainer width="100%" height={210}>
       <BarChart accessibilityLayer data={[{scenario:"Baseline",cost:Number(comparison.baseline.estimated_cost_usd),display:comparison.baseline.display.estimated_cost_usd},{scenario:"Optimized",cost:Number(comparison.optimized.estimated_cost_usd),display:comparison.optimized.display.estimated_cost_usd}]} margin={{top:12,right:16,left:0,bottom:8}}>
         <CartesianGrid vertical={false} stroke="#dce5df" strokeDasharray="3 4"/>
         <XAxis dataKey="scenario" tick={chartTick} axisLine={false} tickLine={false}/>
         <YAxis tick={chartTick} axisLine={false} tickLine={false} width={70}/>
         <Tooltip contentStyle={tooltipStyle} formatter={(_value,_name,item)=>item.payload.display??"Unavailable"}/>
         <Bar name="Estimated USD" dataKey="cost" fill="#90ac42" radius={[5,5,0,0]} isAnimationActive={!reducedMotion} animationDuration={220}/>
       </BarChart>
     </ResponsiveContainer>:<p className="empty">The cost chart requires complete estimates for both scenarios. Known subtotals appear below.</p>}
   </Panel>
 </div>
 <div className="tiles">
   {bench("Baseline generation calls",comparison.baseline.generation_calls)}
   {bench("Optimized generation calls",comparison.optimized.generation_calls)}
   {bench("Safe generation calls avoided",comparison.safe_calls_avoided)}
   {bench("Unsafe cache hits",comparison.cache_outcomes.false_hit)}
 </div>
 <div className="tiles">
   {bench("Baseline known subtotal",comparison.baseline.display.known_cost_subtotal_usd)}
   {bench("Optimized known subtotal",comparison.optimized.display.known_cost_subtotal_usd)}
   {bench("Escalation expenditure",comparison.escalation.display.estimated_cost_usd)}
   {bench("Embedding expenditure",comparison.embedding.display.estimated_cost_usd)}
 </div>
 <p className="context">Unpriced requests · baseline {comparison.baseline.unknown_cost_requests}, optimized {comparison.optimized.unknown_cost_requests}.
   {" "}Escalation known subtotal {comparison.escalation.display.known_cost_subtotal_usd}.
   {" "}Embedding known subtotal {comparison.embedding.display.known_cost_subtotal_usd} includes lookup and storage.</p>
 <div className="tiles">
   {bench("Input token cost",comparison.optimized.components.input_cost_usd.display.estimated_cost_usd)}
   {bench("Cached input cost",comparison.optimized.components.cached_input_cost_usd.display.estimated_cost_usd)}
   {bench("Output token cost",comparison.optimized.components.output_cost_usd.display.estimated_cost_usd)}
 </div>
 <p className="context">Input costs include embeddings. Cached input is priced separately. Output costs include thinking tokens. Missing components remain unavailable.</p>
 <div className="panels">
   {comparison.by_model.map(row=><Panel key={row.model} title={row.model} note="Synthetic benchmark · model costs in USD">
     {[["Baseline",row.baseline],["Optimized",row.optimized]].map(([name,metrics])=>{
       const metric=metrics as typeof row.baseline;
       return <div key={String(name)}>
         <p className="line"><span>{String(name)}</span><strong>{metric.display.estimated_cost_usd??"Unavailable"}</strong></p>
         <p className="context">Known subtotal {metric.display.known_cost_subtotal_usd} · {metric.unknown_cost_requests} of {metric.requests} requests unpriced</p>
       </div>;
     })}
   </Panel>)}
 </div>
 <details className="panel"><summary>Comparison methodology</summary>
   <p className="context">Both scenarios process the same ordered routing fixtures. The baseline calls the large role once per query.
     Each role uses its own synthetic response and token-usage fixture, including thinking tokens.
     The optimized run uses the existing cache, router, evaluator and escalation pipeline at threshold {comparison.threshold}.
     Query and response eligibility are explicitly reviewed.</p>
   <p className="context">Amounts use the saved benchmark pricing and recorded attempts. Embedding costs include lookup and storage.
     Missing prices or usage withhold totals. False or unreviewed cache hits withhold savings.
     Synthetic vectors and responses cannot establish production quality or savings.</p>
 </details>
 </>}
 <h2 className="metric-group">Historical expenditure</h2>
 <div className="tiles">{tile("Total estimated expenditure",costs?.display.estimated_cost_usd,"Historical estimate")}{tile("Cost per request",costs?.display.average_cost_per_query_usd,"Historical estimate")}{tile("Input token cost",costs?.components.input_cost_usd.display.estimated_cost_usd,"Historical estimate")}{tile("Output token cost",costs?.components.output_cost_usd.display.estimated_cost_usd,"Historical estimate")}</div>
 <div className="tiles">{tile("Cached input cost",costs?.components.cached_input_cost_usd.display.estimated_cost_usd,"Historical estimate")}{tile("Embedding expenditure",costs?.embedding.display.estimated_cost_usd,"Historical estimate")}{tile("Known cost subtotal",costs?.display.known_cost_subtotal_usd,"Historical estimate")}{tile("Requests with unknown cost",costs?.unknown_cost_requests,"Historical")}</div>
 <div className="panels">{chart(true)}<Panel title="Historical cost by model" note={"Historical estimates · "+currency}>{models.filter(m=>model==="All models"||model===m).map(m=><p className="line" key={m}>{m}<b>{costs?.by_model.find(item=>item.model===m)?.display.estimated_cost_usd??"Unavailable"}</b></p>)}</Panel></div>
 <p className="context">Older records may lack token cost breakdowns. Historical savings are unavailable because request logs lack paired baseline usage and linked cache outcomes. Live measurements require a connected request source. Benchmark values use USD.</p>
 </>}
 {page===4&&(()=>{const p=b?.performance;const time=(v:number|null|undefined)=>v==null?null:v.toFixed(3)+" ms";const q=(v:number|null|undefined)=>v==null?null:(v*100).toFixed(1)+"%";
 return <><p className="context">Synthetic benchmark · local mock SDK execution in milliseconds, not Gemini provider latency.</p>
 <h2 className="metric-group">Paired end-to-end latency</h2>
 <div className="tiles">{["baseline","optimized"].flatMap(k=>["average_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms"].map(f=>bench(k[0].toUpperCase()+k.slice(1)+" "+(f==="average_ms"?"average":f.replace("_ms","").toUpperCase()),time(p?.[k]?.[f]))))}</div>
 <div className="tiles">{bench(p?.improvement_ms!=null&&p.improvement_ms<0?"Latency regression":"Latency improvement",time(p?.improvement_ms==null?null:Math.abs(p.improvement_ms)))}{bench("Safe cache-hit latency",time(p?.cache_hit?.average_ms))}{[["cache_lookup","Cache lookup"],["embedding","Embedding SDK"],["small_generation","Small generation"],["large_generation","Large generation"],["escalation","Escalation overhead"]].map(([k,l])=>bench(l,time(p?.[k]?.average_ms)))}</div>
 <p className="context">Cache lookup includes local embedding, SQLite and search. Component SDK timings are contained in end-to-end latency and must not be added to it. Incomplete or unsafe pairs withhold improvement.</p>
 <h2 className="metric-group">Heuristic answer quality</h2>
 <div className="tiles">{bench("Baseline pass rate",q(p?.baseline_quality?.pass_rate))}{bench("Optimized pass rate",q(p?.optimized_quality?.pass_rate))}{bench("Initial acceptance",q(p?.initial_quality?.pass_rate))}{bench("Escalation success",q(p?.escalation_quality?.pass_rate))}</div>
 <div className="tiles">{bench("Evaluated answers",p?.optimized_quality?.total)}{bench("Passed answers",p?.optimized_quality?.passed)}{bench("Failed answers",p?.optimized_quality?.failed)}{bench("Escalations",p?.escalation_count)}</div>
 <p className="context">Synthetic responses; heuristic scores do not verify factual correctness.</p>
 <h2 className="metric-group">Historical request latency</h2>
 <div className="tiles">{tile("Average latency",ms(avg(lat)))}{[50,95,99].map(x=>tile("P"+x+" latency",ms(pct(x/100))))}{tile("Input tokens",sum(rows.map(r=>r.input)))}{tile("Output tokens",sum(rows.map(r=>r.output)))}</div></>})()}
 {page===5&&<div className="tiles system-list">{tile("Small Gemini model",data?.small,"Configured")}{tile("Large Gemini model",data?.large,"Configured")}{tile("Cache schema",data?.schema,"Configuration")}{tile("Application cache version",data?.version,"Configuration")}{tile("Similarity threshold",data?.threshold,"Default")}{tile("Evaluation threshold",data?.evaluation,"Default")}{tile("Database status","Not probed","Local")}{tile("API connectivity","Not probed","Live")}{unavailable("Application version")}{tile("Last successful request",rows.find(r=>r.status==="success")?.time)}</div>}
 </div>
 <div className="sources" aria-label="Data sources"><span>Live · disconnected</span><span data-ready={data?.hs==="Ready"}>Historical · {data?.hs??"Waiting"}</span><span data-ready={data?.bs==="Ready"}>Synthetic benchmark · {data?.bs??"Waiting"}</span></div>
 <p className="context" role="status">{error|| (invalid?"Choose a valid date range.":currency!=="USD"&&(!Number.isFinite(factor)||factor<=0)?"Enter a positive conversion rate.":"Date and model filters apply to historical requests. Benchmark results cover the full offline run.")}</p>
 </div></div></main>
}
