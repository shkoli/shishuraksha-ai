import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Dot } from 'recharts';
import { Download, Share2, AlertTriangle, Phone, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { CASES, RISK_COLORS } from '../mockData';
import RiskBadge from '../components/RiskBadge';
import Footer from '../components/Footer';

export default function Reports() {
  const [selected, setSelected] = useState(CASES[0].id);
  const c = CASES.find(x=>x.id===selected) || CASES[0];
  const modalities = [
    { name:'SDQ Score', score: Math.round((c.sdq/22)*100), color:'#6366f1' },
    { name:'CPSS Score', score: Math.round((c.cpss/85)*100), color:'#8b5cf6' },
    { name:'Narrative', score: c.narrative, color:'#ec4899' },
    { name:'Image Features', score: c.image, color:'#06b6d4' },
  ];
  const drivers = [
    { name:'SDQ', value: modalities[0].score },
    { name:'CPSS', value: modalities[1].score },
    { name:'Narrative', value: modalities[2].score },
    { name:'Image', value: modalities[3].score },
  ];
  const riskC = RISK_COLORS[c.risk];

  // Compute trend from screeningHistory if present
  const history = c.screeningHistory;
  let trend = null;
  if (history && history.length >= 2) {
    const delta = history[history.length - 1].score - history[history.length - 2].score;
    if (delta > 0.10) trend = 'worsening';
    else if (delta < -0.10) trend = 'improving';
    else trend = 'stable';
  }

  const TREND_META = {
    worsening: { icon: TrendingUp,   color: '#ef4444', bg: '#fef2f2', border: '#fecaca', label: 'Risk increasing' },
    improving: { icon: TrendingDown, color: '#22c55e', bg: '#f0fdf4', border: '#bbf7d0', label: 'Risk decreasing' },
    stable:    { icon: Minus,        color: '#94a3b8', bg: '#f8fafc', border: '#e2e8f0', label: 'Stable'          },
  };

  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:20 }}>
        <h1 style={{ fontSize:22, fontWeight:700 }}>Case Report</h1>
        <div style={{ display:'flex', gap:10, alignItems:'center' }}>
          <select value={selected} onChange={e=>setSelected(e.target.value)} style={{ padding:'8px 12px', border:'1px solid #e2e8f0', borderRadius:8, fontSize:13, outline:'none', background:'white' }}>
            {CASES.map(x=><option key={x.id} value={x.id}>{x.id}</option>)}
          </select>
          <button style={{ display:'flex', alignItems:'center', gap:6, padding:'8px 16px', background:'#6366f1', color:'white', border:'none', borderRadius:8, fontSize:13, fontWeight:600, cursor:'pointer' }}>
            <Download size={15}/> PDF
          </button>
          <button style={{ display:'flex', alignItems:'center', gap:6, padding:'8px 16px', background:'white', color:'#374151', border:'1px solid #e2e8f0', borderRadius:8, fontSize:13, cursor:'pointer' }}>
            <Share2 size={15}/> Share
          </button>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:16 }}>
        {/* Child Profile */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:700, marginBottom:14, color:'#6366f1' }}>Child Profile</h3>
          {[['Case ID', c.id],['Age', `${c.age} years`],['Gender', c.gender],['Division', c.division],['Date', c.date]].map(([k,v])=>(
            <div key={k} style={{ display:'flex', justifyContent:'space-between', padding:'8px 0', borderBottom:'1px solid #f1f5f9', fontSize:14 }}>
              <span style={{ color:'#64748b', fontWeight:500 }}>{k}</span>
              <span style={{ fontWeight:600, color:'#0f172a' }}>{v}</span>
            </div>
          ))}
          <div style={{ marginTop:12 }}><RiskBadge risk={c.risk} confidence={c.confidence}/></div>
        </div>

        {/* Modality Scores */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:700, marginBottom:14 }}>Modality Scores</h3>
          {modalities.map(m=>(
            <div key={m.name} style={{ marginBottom:14 }}>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5, fontSize:13 }}>
                <span style={{ fontWeight:600, color:'#374151' }}>{m.name}</span>
                <span style={{ fontWeight:700, color:m.color }}>{m.score}%</span>
              </div>
              <div style={{ background:'#e2e8f0', borderRadius:99, height:8 }}>
                <div style={{ background:m.color, width:`${m.score}%`, height:8, borderRadius:99, transition:'width 0.5s ease' }}/>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:16 }}>
        {/* Risk Drivers chart */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:700, marginBottom:14 }}>Risk Driver Analysis</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={drivers}>
              <XAxis dataKey="name" tick={{ fontSize:12 }}/>
              <YAxis domain={[0,100]} tick={{ fontSize:12 }}/>
              <Tooltip/>
              <Bar dataKey="value" name="Score" radius={[4,4,0,0]}>
                {drivers.map((d,i)=>(
                  <Cell key={i} fill={i===0?'#6366f1': i===1?'#8b5cf6': i===2?'#ec4899':'#06b6d4'}/>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Referral Decision */}
        <div style={{ background: riskC.bg, border:`1px solid ${riskC.border}`, borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:700, marginBottom:14, color: riskC.text }}>Referral Decision</h3>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:14 }}>
            <div style={{ width:48, height:48, borderRadius:'50%', background:`${riskC.dot}20`, border:`2px solid ${riskC.dot}`, display:'flex', alignItems:'center', justifyContent:'center' }}>
              <span style={{ fontWeight:700, fontSize:16, color:riskC.dot }}>{c.score}%</span>
            </div>
            <div>
              <p style={{ fontWeight:700, color:riskC.text }}>{c.risk} Risk Level</p>
              <p style={{ fontSize:13, color:'#64748b' }}>Overall risk score</p>
            </div>
          </div>
          {(c.risk==='Critical'||c.risk==='High') && (
            <div>
              <p style={{ fontSize:13, fontWeight:600, color:'#374151', marginBottom:10 }}>Immediate Referral:</p>
              <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
                <a href="tel:1098" style={{ display:'flex', alignItems:'center', gap:6, background:'#ef4444', color:'white', borderRadius:8, padding:'7px 14px', textDecoration:'none', fontSize:13, fontWeight:600 }}><Phone size={13}/> DSS 1098</a>
                <a href="tel:16767" style={{ display:'flex', alignItems:'center', gap:6, background:'#6366f1', color:'white', borderRadius:8, padding:'7px 14px', textDecoration:'none', fontSize:13, fontWeight:600 }}><Phone size={13}/> OCC 16767</a>
              </div>
            </div>
          )}
          {(c.risk==='Moderate'||c.risk==='Low') && (
            <p style={{ fontSize:13, color:'#64748b' }}>Continue monitoring. Schedule follow-up screening in 4 weeks.</p>
          )}
        </div>
      </div>

      {/* Screening History */}
      {history && history.length > 0 && (() => {
        const tm = TREND_META[trend];
        const TrendIcon = tm.icon;
        const chartData = history.map(h => ({ date: h.date, score: Math.round(h.score * 100), risk: h.risk, confidence: h.confidence }));
        return (
          <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20, marginBottom:16 }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:14 }}>
              <h3 style={{ fontSize:15, fontWeight:700 }}>Screening History</h3>
              <div style={{ display:'flex', alignItems:'center', gap:6, padding:'5px 12px', background: tm.bg, border:`1px solid ${tm.border}`, borderRadius:99 }}>
                <TrendIcon size={14} color={tm.color}/>
                <span style={{ fontSize:13, fontWeight:600, color: tm.color }}>{tm.label}</span>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData} margin={{ top:8, right:16, left:0, bottom:8 }}>
                <XAxis dataKey="date" tick={{ fontSize:11 }} tickLine={false}/>
                <YAxis domain={[0,100]} tick={{ fontSize:11 }} tickLine={false} axisLine={false} unit="%"/>
                <Tooltip formatter={(v) => [`${v}%`, 'Risk Score']}/>
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={<Dot r={5} fill="#6366f1" stroke="white" strokeWidth={2}/>}
                  activeDot={{ r:7 }}
                />
              </LineChart>
            </ResponsiveContainer>

            <div style={{ display:'flex', gap:8, marginTop:14, flexWrap:'wrap' }}>
              {history.map((h, i) => {
                const rc = RISK_COLORS[h.risk] || RISK_COLORS.Low;
                return (
                  <div key={i} style={{ flex:'1 1 120px', background:'#f8fafc', border:'1px solid #e2e8f0', borderRadius:10, padding:'10px 12px' }}>
                    <p style={{ fontSize:11, color:'#94a3b8', marginBottom:4 }}>{h.date}</p>
                    <p style={{ fontSize:16, fontWeight:700, color:'#0f172a', marginBottom:4 }}>{Math.round(h.score * 100)}%</p>
                    <span style={{ fontSize:11, fontWeight:600, color: rc.text, background: rc.bg, border:`1px solid ${rc.border}`, borderRadius:99, padding:'2px 8px' }}>{h.risk}</span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

      {/* Bengali disclaimer */}
      <div style={{ background:'#fffbeb', border:'1px solid #fde68a', borderRadius:12, padding:16, display:'flex', gap:12, alignItems:'flex-start' }}>
        <AlertTriangle size={20} color="#d97706" style={{ flexShrink:0, marginTop:2 }}/>
        <div>
          <p style={{ fontWeight:700, color:'#92400e', marginBottom:4 }}>গুরুত্বপূর্ণ দাবিত্যাগ</p>
          <p style={{ fontSize:13, color:'#78350f', lineHeight:1.7 }}>
            এই স্ক্রিনিং রিপোর্টটি একটি AI-সহায়তা টুল দ্বারা তৈরি করা হয়েছে। এটি কোনো চিকিৎসা নির্ণয় নয় এবং প্রশিক্ষিত মানসিক স্বাস্থ্য পেশাদারের মূল্যায়নের বিকল্প নয়। সকল সিদ্ধান্ত একজন যোগ্য পেশাদার দ্বারা নেওয়া উচিত।
          </p>
        </div>
      </div>
      <Footer/>
    </div>
  );
}
