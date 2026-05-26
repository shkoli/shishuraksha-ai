import { AlertTriangle, Users, Activity, CheckCircle, Eye } from 'lucide-react';
import { CASES, RISK_COLORS } from '../mockData';
import RiskBadge from '../components/RiskBadge';
import Footer from '../components/Footer';

const card = (bg, border, iconBg, icon, label, value) => ({ bg, border, iconBg, icon, label, value });
const STAT_CARDS = [
  card('#ede9fe','#ddd6fe','#6366f1', <Activity size={20} color="white"/>, 'Total Screenings', CASES.length),
  card('#fef2f2','#fecaca','#ef4444', <AlertTriangle size={20} color="white"/>, 'Critical Cases', CASES.filter(c=>c.risk==='Critical').length),
  card('#fff7ed','#fed7aa','#f97316', <Users size={20} color="white"/>, 'High Risk', CASES.filter(c=>c.risk==='High').length),
  card('#f0fdf4','#bbf7d0','#22c55e', <CheckCircle size={20} color="white"/>, 'Low Risk', CASES.filter(c=>c.risk==='Low').length),
];

const getBengaliDate = () => {
  const bn = ['জানুয়ারি','ফেব্রুয়ারি','মার্চ','এপ্রিল','মে','জুন','জুলাই','আগস্ট','সেপ্টেম্বর','অক্টোবর','নভেম্বর','ডিসেম্বর'];
  const d = new Date();
  return `${d.getDate()} ${bn[d.getMonth()]} ${d.getFullYear()}`;
};

export default function Dashboard({ onNav }) {
  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom:28 }}>
        <h1 style={{ fontSize:24, fontWeight:700, color:'#0f172a' }}>Welcome back, Koli 👋</h1>
        <p style={{ color:'#64748b', fontSize:14, marginTop:4 }}>{getBengaliDate()}</p>
      </div>

      {/* Stat cards */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16, marginBottom:28 }}>
        {STAT_CARDS.map((c, i) => (
          <div key={i} style={{ background:c.bg, border:`1px solid ${c.border}`, borderRadius:12, padding:20, display:'flex', alignItems:'center', gap:16 }}>
            <div style={{ background:c.iconBg, borderRadius:10, padding:10, display:'flex', flexShrink:0 }}>{c.icon}</div>
            <div>
              <p style={{ fontSize:13, color:'#64748b', marginBottom:2 }}>{c.label}</p>
              <p style={{ fontSize:28, fontWeight:700, color:'#0f172a' }}>{c.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Recent cases table */}
      <div style={{ background:'white', borderRadius:12, border:'1px solid #e2e8f0', overflow:'hidden' }}>
        <div style={{ padding:'18px 24px', borderBottom:'1px solid #e2e8f0', display:'flex', justifyContent:'space-between', alignItems:'center' }}>
          <h2 style={{ fontSize:16, fontWeight:600 }}>Recent Screenings</h2>
          <button onClick={() => onNav('cases')} style={{ fontSize:13, color:'#6366f1', background:'none', border:'none', cursor:'pointer', fontWeight:500 }}>View all →</button>
        </div>
        <table style={{ width:'100%', borderCollapse:'collapse' }}>
          <thead>
            <tr style={{ background:'#f8fafc' }}>
              {['Case ID','Age','Division','Risk Level','Score','Date','Action'].map(h => (
                <th key={h} style={{ padding:'10px 16px', textAlign:'left', fontSize:12, fontWeight:600, color:'#64748b', borderBottom:'1px solid #e2e8f0' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CASES.map((c, i) => (
              <tr key={c.id} style={{ background: i%2===0?'white':'#fafafa', transition:'background 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.background='#f0f9ff'}
                onMouseLeave={e => e.currentTarget.style.background = i%2===0?'white':'#fafafa'}
              >
                <td style={{ padding:'12px 16px', fontSize:13, fontWeight:600, color:'#6366f1' }}>{c.id}</td>
                <td style={{ padding:'12px 16px', fontSize:13 }}>{c.age} yrs</td>
                <td style={{ padding:'12px 16px', fontSize:13 }}>{c.division}</td>
                <td style={{ padding:'12px 16px' }}><RiskBadge risk={c.risk} confidence={c.confidence}/></td>
                <td style={{ padding:'12px 16px', fontSize:13, fontWeight:600 }}>
                  <span style={{ color: c.score>=80?'#ef4444': c.score>=60?'#f97316': c.score>=40?'#eab308':'#22c55e' }}>{c.score}%</span>
                </td>
                <td style={{ padding:'12px 16px', fontSize:13, color:'#64748b' }}>{c.date}</td>
                <td style={{ padding:'12px 16px' }}>
                  <button onClick={() => onNav('reports')} style={{ display:'flex', alignItems:'center', gap:5, background:'#6366f1', color:'white', border:'none', borderRadius:8, padding:'5px 12px', fontSize:12, cursor:'pointer', fontWeight:500 }}>
                    <Eye size={13}/> View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Footer/>
    </div>
  );
}
