import { RISK_COLORS } from '../mockData';

export default function RiskBadge({ risk, confidence }) {
  const c = RISK_COLORS[risk] || RISK_COLORS.Low;
  const label = confidence != null ? `${risk} (${confidence}% confidence)` : risk;
  return (
    <span style={{
      display:'inline-flex', alignItems:'center', gap:5,
      background: c.bg, color: c.text, border:`1px solid ${c.border}`,
      borderRadius:20, padding:'3px 10px', fontSize:12, fontWeight:600,
    }}>
      <span style={{ width:7, height:7, borderRadius:'50%', background: c.dot, flexShrink:0 }} />
      {label}
    </span>
  );
}
