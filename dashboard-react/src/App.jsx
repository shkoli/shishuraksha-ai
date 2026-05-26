import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Screening from './pages/Screening';
import Cases from './pages/Cases';
import Analytics from './pages/Analytics';
import Reports from './pages/Reports';

const PAGES = {
  dashboard: Dashboard,
  screening: Screening,
  cases: Cases,
  analytics: Analytics,
  reports: Reports,
};

export default function App() {
  const [page, setPage] = useState('dashboard');
  const Page = PAGES[page] || Dashboard;

  return (
    <div style={{ display:'flex', minHeight:'100vh', background:'#f8fafc' }}>
      <Sidebar activePage={page} onNav={setPage}/>
      <main style={{ marginLeft:240, flex:1, padding:'32px 32px 0', minWidth:0 }}>
        <Page onNav={setPage}/>
      </main>
    </div>
  );
}
