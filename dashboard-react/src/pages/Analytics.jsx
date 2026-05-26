import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, Legend } from 'recharts';
import { CASES, DIVISION_COUNTS, SCREENING_TREND, RISK_DRIVERS, RISK_COLORS } from '../mockData';
import Footer from '../components/Footer';

const riskDist = ['Critical','High','Moderate','Low'].map(r => ({
  name: r, value: CASES.filter(c=>c.risk===r).length, color: RISK_COLORS[r].dot
}));

const METRIC_CARDS = [
  { label:'Total Screenings', value:15, sub:'This month', color:'#6366f1', bg:'#ede9fe', border:'#ddd6fe' },
  { label:'Avg Risk Score', value:'61%', sub:'Across all cases', color:'#f97316', bg:'#fff7ed', border:'#fed7aa' },
  { label:'Referrals Made', value:7, sub:'DSS/OCC referrals', color:'#ef4444', bg:'#fef2f2', border:'#fecaca' },
  { label:'Cases Resolved', value:3, sub:'Closed this month', color:'#22c55e', bg:'#f0fdf4', border:'#bbf7d0' },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:8, padding:'8px 12px', fontSize:13, boxShadow:'0 4px 12px rgba(0,0,0,0.1)' }}>
        <p style={{ fontWeight:600 }}>{label}</p>
        {payload.map((p,i) => <p key={i} style={{ color:p.color }}>{p.name}: {p.value}</p>)}
      </div>
    );
  }
  return null;
};

export default function Analytics() {
  return (
    <div>
      <h1 style={{ fontSize:22, fontWeight:700, marginBottom:20 }}>Analytics Overview</h1>

      {/* Metric cards */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16, marginBottom:24 }}>
        {METRIC_CARDS.map((c,i)=>(
          <div key={i} style={{ background:c.bg, border:`1px solid ${c.border}`, borderRadius:12, padding:18 }}>
            <p style={{ fontSize:13, color:'#64748b', marginBottom:6 }}>{c.label}</p>
            <p style={{ fontSize:28, fontWeight:700, color:c.color }}>{c.value}</p>
            <p style={{ fontSize:12, color:'#94a3b8', marginTop:4 }}>{c.sub}</p>
          </div>
        ))}
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20, marginBottom:20 }}>
        {/* Pie chart */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:600, marginBottom:16 }}>Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={riskDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({name,value})=>`${name}: ${value}`} labelLine={false}>
                {riskDist.map((entry,i)=><Cell key={i} fill={entry.color}/>)}
              </Pie>
              <Tooltip/>
              <Legend/>
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar chart by division */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:600, marginBottom:16 }}>Cases by Division</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={DIVISION_COUNTS} margin={{ bottom:20 }}>
              <XAxis dataKey="division" tick={{ fontSize:11 }} angle={-30} textAnchor="end"/>
              <YAxis tick={{ fontSize:12 }} allowDecimals={false}/>
              <Tooltip content={<CustomTooltip/>}/>
              <Bar dataKey="count" name="Cases" fill="#6366f1" radius={[4,4,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'2fr 1fr', gap:20 }}>
        {/* Line chart */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:600, marginBottom:16 }}>Screenings — Last 30 Days</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={SCREENING_TREND}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
              <XAxis dataKey="day" tick={{ fontSize:10 }} interval={4}/>
              <YAxis tick={{ fontSize:12 }} allowDecimals={false}/>
              <Tooltip content={<CustomTooltip/>}/>
              <Line type="monotone" dataKey="count" name="Screenings" stroke="#6366f1" strokeWidth={2} dot={false} activeDot={{ r:5 }}/>
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Top risk drivers bar */}
        <div style={{ background:'white', border:'1px solid #e2e8f0', borderRadius:12, padding:20 }}>
          <h3 style={{ fontSize:15, fontWeight:600, marginBottom:16 }}>Top Risk Drivers</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={RISK_DRIVERS} layout="vertical">
              <XAxis type="number" tick={{ fontSize:11 }} domain={[0,100]}/>
              <YAxis type="category" dataKey="name" tick={{ fontSize:11 }} width={90}/>
              <Tooltip content={<CustomTooltip/>}/>
              <Bar dataKey="value" name="Importance" fill="#8b5cf6" radius={[0,4,4,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <Footer/>
    </div>
  );
}
