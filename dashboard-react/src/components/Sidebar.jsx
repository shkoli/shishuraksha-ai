import { Shield, LayoutDashboard, Plus, Users, BarChart3, FileText, Lock } from 'lucide-react';

const NAV = [
  { icon: LayoutDashboard, label: 'Dashboard',     page: 'dashboard' },
  { icon: Plus,            label: 'New Screening', page: 'screening' },
  { icon: Users,           label: 'Cases',         page: 'cases' },
  { icon: BarChart3,       label: 'Analytics',     page: 'analytics' },
  { icon: FileText,        label: 'Reports',       page: 'reports' },
];

export default function Sidebar({ activePage, onNav }) {
  return (
    <aside style={{ width:240, minWidth:240, background:'#0a0f1e', minHeight:'100vh', display:'flex', flexDirection:'column', position:'fixed', left:0, top:0, bottom:0, zIndex:50 }}>
      {/* Logo */}
      <div style={{ padding:'24px 20px 16px' }}>
        <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:4 }}>
          <div style={{ background:'#6366f1', borderRadius:10, padding:8, display:'flex' }}>
            <Shield size={20} color="white" />
          </div>
          <span style={{ color:'white', fontWeight:700, fontSize:15 }}>ShishuRaksha AI</span>
        </div>
        <p style={{ color:'#64748b', fontSize:12, marginLeft:38 }}>শিশু সুরক্ষা স্ক্রিনিং</p>
        {/* Gradient accent bar */}
        <div style={{ marginTop:16, height:3, borderRadius:2, background:'linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899)' }} />
      </div>

      {/* Nav */}
      <nav style={{ flex:1, padding:'8px 12px' }}>
        {NAV.map(({ icon: Icon, label, page }) => {
          const active = activePage === page;
          return (
            <button key={page} onClick={() => onNav(page)} style={{
              width:'100%', display:'flex', alignItems:'center', gap:10,
              padding:'10px 12px', borderRadius:8, border:'none', cursor:'pointer',
              background: active ? '#6366f1' : 'transparent',
              color: active ? 'white' : '#94a3b8',
              fontFamily:'Inter,sans-serif', fontSize:14, fontWeight:500,
              marginBottom:2, transition:'all 0.2s ease',
              textAlign:'left',
            }}
            onMouseEnter={e => { if (!active) e.currentTarget.style.background = '#1e293b'; }}
            onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}
            >
              <Icon size={18} />
              {label}
            </button>
          );
        })}
      </nav>

      {/* Bottom */}
      <div style={{ padding:'16px 16px', borderTop:'1px solid #1e293b' }}>
        <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:12 }}>
          <div style={{ width:36, height:36, borderRadius:'50%', background:'linear-gradient(135deg,#6366f1,#8b5cf6)', display:'flex', alignItems:'center', justifyContent:'center', color:'white', fontWeight:700, fontSize:14, flexShrink:0 }}>S</div>
          <div>
            <p style={{ color:'white', fontSize:12, fontWeight:600, lineHeight:1.3 }}>Salma Hoque</p>
            <p style={{ color:'#64748b', fontSize:11 }}>Talukdar Koli</p>
          </div>
        </div>
        <p style={{ color:'#475569', fontSize:11, marginBottom:6 }}>ShishuRaksha AI v1.0.0</p>
        <div style={{ display:'flex', alignItems:'center', gap:6, color:'#475569' }}>
          <Lock size={11} />
          <span style={{ fontSize:11 }}>AES-256-GCM Encrypted</span>
        </div>
      </div>
    </aside>
  );
}
