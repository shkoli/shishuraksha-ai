import { useState } from 'react';
import { Search, Eye } from 'lucide-react';
import { CASES, DIVISIONS } from '../mockData';
import RiskBadge from '../components/RiskBadge';
import Footer from '../components/Footer';

export default function Cases({ onNav }) {
  const [search, setSearch] = useState('');
  const [riskF, setRiskF] = useState('All');
  const [divF, setDivF] = useState('All');

  const filtered = CASES.filter(c =>
    (c.id.toLowerCase().includes(search.toLowerCase()) || c.division.toLowerCase().includes(search.toLowerCase())) &&
    (riskF==='All' || c.risk===riskF) &&
    (divF==='All' || c.division===divF)
  );

  const selStyle = { padding:'8px 12px', border:'1px solid #e2e8f0', borderRadius:8, fontSize:13, outline:'none', background:'white', cursor:'pointer' };

  return (
    <div>
      <h1 style={{ fontSize:22, fontWeight:700, marginBottom:20 }}>Case Management</h1>

      {/* Filters */}
      <div style={{ display:'flex', gap:12, marginBottom:24, flexWrap:'wrap' }}>
        <div style={{ position:'relative', flex:1, minWidth:200 }}>
          <Search size={16} color="#94a3b8" style={{ position:'absolute', left:12, top:'50%', transform:'translateY(-50%)' }}/>
          <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search case ID or division..." style={{ width:'100%', padding:'9px 12px 9px 36px', border:'1px solid #e2e8f0', borderRadius:8, fontSize:14, outline:'none' }}/>
        </div>
        <select style={selStyle} value={riskF} onChange={e=>setRiskF(e.target.value)}>
          <option value="All">All Risk Levels</option>
          {['Critical','High','Moderate','Low'].map(r=><option key={r}>{r}</option>)}
        </select>
        <select style={selStyle} value={divF} onChange={e=>setDivF(e.target.value)}>
          <option value="All">All Divisions</option>
          {DIVISIONS.map(d=><option key={d}>{d}</option>)}
        </select>
      </div>

      <p style={{ fontSize:13, color:'#64748b', marginBottom:16 }}>Showing {filtered.length} of {CASES.length} cases</p>

      {/* Cards grid */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:16 }}>
        {filtered.map(c=>(
          <div key={c.id} style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20, transition:'all 0.2s' }}
            onMouseEnter={e=>{ e.currentTarget.style.boxShadow='0 4px 16px rgba(99,102,241,0.12)'; e.currentTarget.style.borderColor='#6366f1'; }}
            onMouseLeave={e=>{ e.currentTarget.style.boxShadow=''; e.currentTarget.style.borderColor='#e2e8f0'; }}
          >
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:14 }}>
              <span style={{ fontWeight:700, color:'#6366f1', fontSize:14 }}>{c.id}</span>
              <RiskBadge risk={c.risk} confidence={c.confidence}/>
            </div>
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8, marginBottom:16 }}>
              {[['Age', `${c.age} years`],['Gender', c.gender],['Division', c.division],['Date', c.date]].map(([k,v])=>(
                <div key={k}>
                  <p style={{ fontSize:11, color:'#94a3b8', fontWeight:600, textTransform:'uppercase', letterSpacing:'0.05em' }}>{k}</p>
                  <p style={{ fontSize:13, fontWeight:500, color:'#374151', marginTop:2 }}>{v}</p>
                </div>
              ))}
            </div>
            <div style={{ marginBottom:14 }}>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4, fontSize:12 }}>
                <span style={{ color:'#64748b' }}>Risk Score</span>
                <span style={{ fontWeight:700, color: c.score>=80?'#ef4444': c.score>=60?'#f97316': c.score>=40?'#eab308':'#22c55e' }}>{c.score}%</span>
              </div>
              <div style={{ background:'#e2e8f0', borderRadius:99, height:6 }}>
                <div style={{ width:`${c.score}%`, height:6, borderRadius:99, background: c.score>=80?'#ef4444': c.score>=60?'#f97316': c.score>=40?'#eab308':'#22c55e' }}/>
              </div>
            </div>
            <button onClick={()=>onNav('reports')} style={{ width:'100%', display:'flex', alignItems:'center', justifyContent:'center', gap:6, background:'#6366f1', color:'white', border:'none', borderRadius:8, padding:'8px', fontSize:13, fontWeight:600, cursor:'pointer' }}>
              <Eye size={14}/> View Report
            </button>
          </div>
        ))}
      </div>
      <Footer/>
    </div>
  );
}
