"""Validated projections of paired synthetic timings and heuristic decisions."""
import math

def latency_stats(samples):
    if not samples:
        return dict(count=0, average_ms=None, p50_ms=None, p95_ms=None, p99_ms=None, min_ms=None, max_ms=None)
    if any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("Latency samples must be finite nonnegative milliseconds.")
    ordered=sorted(samples)
    def percentile(p):
        return ordered[max(0, math.ceil(len(ordered)*p)-1)]
    return dict(count=len(ordered), average_ms=sum(ordered)/len(ordered),
                p50_ms=percentile(.5), p95_ms=percentile(.95), p99_ms=percentile(.99),
                min_ms=ordered[0], max_ms=ordered[-1])

def quality_stats(decisions):
    if not decisions:
        return dict(total=0, passed=0, failed=0, pass_rate=None)
    if any(x not in ("pass","fail") for x in decisions):
        raise ValueError("Invalid quality decision.")
    passed=decisions.count("pass")
    return dict(total=len(decisions), passed=passed, failed=len(decisions)-passed,
                pass_rate=passed/len(decisions))

def project_latency_quality(ledger):
    """Return unavailable for old ledgers; never substitute missing timings."""
    if not isinstance(ledger,dict) or not isinstance(ledger.get("queries"),list):
        return None
    rows=ledger["queries"]
    if any(not isinstance(row,dict) or not isinstance(row.get("timing"),dict)
           or not isinstance(row.get("quality"),dict) for row in rows):
        return None
    try:
        def samples(scenario,field):
            values=[row["timing"][scenario][field] for row in rows]
            return latency_stats(values) if all(v is not None for v in values) else latency_stats([])
        baseline=samples("baseline","end_to_end_ms")
        optimized=samples("optimized","end_to_end_ms")
        def components(scenario,key):
            return latency_stats([r["timing"][scenario][key] for r in rows
                                  if r["timing"][scenario][key] is not None])
        baseq=quality_stats([r["quality"]["baseline"] for r in rows])
        optq=quality_stats([r["quality"]["optimized"] for r in rows])
        initial=quality_stats([r["quality"]["initial"] for r in rows if r["quality"]["initial"] is not None])
        escalated=[r for r in rows if r["timing"]["optimized"]["escalation_ms"] is not None]
        success=quality_stats([r["quality"]["optimized"] for r in escalated])
        comparable=baseline["count"]==optimized["count"]==len(rows) and all(
            r.get("cache_outcome") not in ("false_hit","unreviewed_hit") for r in rows)
        improvement=(baseline["average_ms"]-optimized["average_ms"]) if comparable and rows else None
        return dict(source="synthetic mock execution; not Gemini network latency",
                    baseline=baseline,optimized=optimized,
                    cache_lookup=components("optimized","cache_lookup_ms"),
                    embedding=components("optimized","embedding_ms"),
                    small_generation=components("optimized","small_generation_ms"),
                    large_generation=components("optimized","large_generation_ms"),
                    escalation=components("optimized","escalation_ms"),
                    cache_hit=latency_stats([r["timing"]["optimized"]["end_to_end_ms"] for r in rows if r["cache_outcome"]=="safe_hit"]),
                    baseline_quality=baseq, optimized_quality=optq,initial_quality=initial,
                    escalation_quality=success,escalation_count=len(escalated),
                    improvement_ms=improvement,comparable=comparable)
    except (KeyError,TypeError,ValueError,ArithmeticError):
        return None
