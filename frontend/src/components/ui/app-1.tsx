import {useState,useSyncExternalStore,type ReactNode} from "react";
import {Search,SlidersHorizontal,ArrowUpRight,Sparkles,DatabaseZap,Activity,Database,RefreshCw,Route,Wallet,Settings,LayoutDashboard} from "lucide-react";
import {AreaChart,Area,LineChart,Line,CartesianGrid,XAxis,YAxis,Tooltip,ResponsiveContainer} from "recharts";
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
 const {data,loading,error,refresh}=useWorkspace(); const [page,P]=useState(0),[range,R]=useState("7 days"),[model,M]=useState("All models"),[currency,C]=useState("USD"),[rate,F]=useState("1"),[from,A]=useState(day(new Date())),[to,Z]=useState(day(new Date())),[threshold,T]=useState<number|null>(null);
 const today=day(new Date()),start=range==="Custom"?from:day(new Date(Date.now()-86400000*(range==="Today"?0:range==="7 days"?6:29))),end=range==="Custom"?to:today;
 const invalid=!start||!end||start>end,rows=(data?.history??[]).filter(r=>!invalid&&r.time.slice(0,10)>=start&&r.time.slice(0,10)<=end&&(model==="All models"||r.model===model)).sort((a,b)=>b.time.localeCompare(a.time));
 const b=data?.bench,c=b?.cache.find(x=>x.t===(threshold??data?.threshold))??b?.cache[0],n=data?.history&&!invalid?rows.length:null;
 const factor=currency==="USD"?1:Number(rate), money=(v:number|null)=>v==null||!Number.isFinite(factor)||factor<=0?null:new Intl.NumberFormat("en",{style:"currency",currency,maximumFractionDigits:6}).format(v*factor);
 const total=sum(rows.map(r=>r.cost)),lat=rows.flatMap(r=>r.ms==null?[]:[r.ms]).sort((a,b)=>a-b),pct=(p:number)=>lat.length?lat[Math.ceil(lat.length*p)-1]:null;
 const models=[...new Set([data?.small,data?.large,...(data?.history??[]).map(r=>r.model)].filter((x):x is string=>!!x))];
 const daily=new Map<string,RequestRow[]>();for(const r of rows){const d=r.time.slice(0,10);daily.set(d,[...(daily.get(d)??[]),r])}
 const trend=[...daily].sort(([a],[b])=>a.localeCompare(b)).map(([date,rs])=>({date,requests:rs.length,cost:sum(rs.map(r=>r.cost))}));
 const tile=(l:string,v:ReactNode,s="Historical")=><Tile key={l} label={l} value={v} source={s}/>;
 const bench=(l:string,v:ReactNode)=>tile(l,v,"Benchmark"); const unavailable=(l:string)=>tile(l,null,"Unmeasured");
 const convertedTrend = trend.map(point => ({
   ...point, cost: point.cost != null && Number.isFinite(factor) && factor > 0 ? point.cost * factor : null,
 }));
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
           <Tooltip contentStyle={tooltipStyle}/>
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
 <div className="four surface">{tile("Total requests",n)}{tile("Estimated API cost",money(total),"Historical estimate")}{bench("Cache hit rate",percent(c?.hitRate))}{unavailable("Estimated cost avoided")}</div>
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
 {page===3&&<><div className="tiles">{tile("Total estimated expenditure",money(total),"Historical estimate")}{tile("Cost per request",money(total!=null&&rows.length?total/rows.length:null),"Historical estimate")}{unavailable("Input token cost")}{unavailable("Output token cost")}</div><h2 className="metric-group">Cost comparison</h2><div className="tiles">{unavailable("Baseline cost")}{unavailable("Optimized scenario cost")}{unavailable("Estimated savings")}{unavailable("Savings percentage")}</div><div className="panels">{chart(true)}<Panel title="Cost by model" note={"Historical estimates · "+currency}>{models.filter(m=>model==="All models"||model===m).map(m=><p className="line" key={m}>{m}<b>{money(sum(rows.filter(r=>r.model===m).map(r=>r.cost)))??"Unavailable"}</b></p>)}</Panel></div></>}
 {page===4&&<><h2 className="metric-group">Request latency</h2><div className="tiles">{tile("Average latency",ms(avg(lat)))}{[50,95,99].map(p=>tile("P"+p+" latency",ms(pct(p/100))))}</div><h2 className="metric-group">Processing time</h2><div className="tiles">{bench("Cache lookup latency",ms(c?.lookup))}{["Embedding","Generation"].map((k)=>tile(k+" latency",ms(avg(rows.filter(r=>(r.kind==="embedding")===(k==="Embedding")&&r.ms!=null).map(r=>r.ms!)))))}{unavailable("Escalation overhead")}</div><h2 className="metric-group">Usage and quality</h2><div className="tiles">{tile("Input tokens",sum(rows.map(r=>r.input)))}{tile("Output tokens",sum(rows.map(r=>r.output)))}{bench("Final answer-quality pass rate",ratio(b?.passed,b?.queries))}</div></>}
 {page===5&&<div className="tiles system-list">{tile("Small Gemini model",data?.small,"Configured")}{tile("Large Gemini model",data?.large,"Configured")}{tile("Cache schema",data?.schema,"Configuration")}{tile("Application cache version",data?.version,"Configuration")}{tile("Similarity threshold",data?.threshold,"Default")}{tile("Evaluation threshold",data?.evaluation,"Default")}{tile("Database status","Not probed","Local")}{tile("API connectivity","Not probed","Live")}{unavailable("Application version")}{tile("Last successful request",rows.find(r=>r.status==="success")?.time)}</div>}
 </div>
 <div className="sources" aria-label="Data sources"><span>Live · disconnected</span><span data-ready={data?.hs==="Ready"}>Historical · {data?.hs??"Waiting"}</span><span data-ready={data?.bs==="Ready"}>Synthetic benchmark · {data?.bs??"Waiting"}</span></div>
 <p className="context" role="status">{error|| (invalid?"Choose a valid date range.":currency!=="USD"&&factor<=0?"Enter a positive conversion rate.":"Date and model filters apply to historical requests. Benchmark results cover the full offline run.")}</p>
 </div></div></main>
}
