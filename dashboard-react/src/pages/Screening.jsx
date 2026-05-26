import { useState, useRef } from 'react';
import { Upload, CheckCircle, Phone } from 'lucide-react';
import Footer from '../components/Footer';

const DIVISIONS = ['Dhaka','Chittagong','Rajshahi','Khulna','Barishal','Sylhet','Rangpur','Mymensingh'];
const SDQ_ITEMS = [
  { en:'Emotional Symptoms', bn:'আবেগীয় লক্ষণ' },
  { en:'Conduct Problems', bn:'আচরণগত সমস্যা' },
  { en:'Hyperactivity', bn:'অতিসক্রিয়তা' },
  { en:'Peer Problems', bn:'বন্ধু সম্পর্কিত সমস্যা' },
  { en:'Prosocial Behaviour', bn:'সামাজিক আচরণ' },
];
const SAMPLE_PROMPTS = [
  'শিশুটি প্রায়ই কাঁদে এবং একা থাকতে চায়।',
  'সে স্কুলে যেতে ভয় পায় এবং ঘুমের সমস্যা আছে।',
  'পরিবারের সাথে স্বাভাবিক যোগাযোগ বন্ধ হয়ে গেছে।',
];

const STEP_LABELS = ['Case Info','SDQ & CPSS','Narrative','Image Upload','AI Result'];

export default function Screening() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState({ age:10, gender:'Female', division:'Dhaka', area:'Urban', sdq:[5,5,5,5,5], cpss:30, narrative:'', images:[] });
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const caseId = 'SR-2024-' + String(Math.floor(Math.random()*900)+100).padStart(3,'0');
  const fileRef = useRef();

  const sdqTotal = form.sdq.reduce((a,b)=>a+b,0);
  const overallScore = Math.round((sdqTotal/50)*40 + (form.cpss/85)*35 + (form.narrative.length/500)*25);
  const risk = overallScore>=80?'Critical': overallScore>=60?'High': overallScore>=40?'Moderate':'Low';
  const riskColor = { Critical:'#ef4444', High:'#f97316', Moderate:'#eab308', Low:'#22c55e' }[risk];

  const handleSubmit = () => {
    setLoading(true);
    setTimeout(() => { setLoading(false); setDone(true); }, 2500);
  };

  const inputStyle = { width:'100%', padding:'9px 12px', border:'1px solid #e2e8f0', borderRadius:8, fontSize:14, fontFamily:'Inter,sans-serif', outline:'none' };
  const labelStyle = { fontSize:13, fontWeight:600, color:'#374151', display:'block', marginBottom:6 };

  return (
    <div style={{ maxWidth:720, margin:'0 auto' }}>
      <h1 style={{ fontSize:22, fontWeight:700, marginBottom:20 }}>New Screening</h1>

      {/* Progress bar */}
      <div style={{ marginBottom:28 }}>
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
          {STEP_LABELS.map((l,i) => (
            <span key={i} style={{ fontSize:12, fontWeight: i<=step?600:400, color: i<=step?'#6366f1':'#94a3b8' }}>{l}</span>
          ))}
        </div>
        <div style={{ background:'#e2e8f0', borderRadius:99, height:6 }}>
          <div style={{ background:'#6366f1', height:6, borderRadius:99, width:`${((step+1)/5)*100}%`, transition:'width 0.3s ease' }} />
        </div>
      </div>

      <div style={{ background:'white', borderRadius:12, border:'1px solid #e2e8f0', padding:28 }}>
        {/* Step 1 */}
        {step===0 && (
          <div>
            <h2 style={{ fontSize:16, fontWeight:600, marginBottom:20 }}>Case Information</h2>
            <div style={{ marginBottom:16 }}>
              <label style={labelStyle}>Case ID (Auto-generated)</label>
              <input style={{ ...inputStyle, background:'#f8fafc', color:'#6366f1', fontWeight:600 }} value={caseId} readOnly/>
            </div>
            <div style={{ marginBottom:16 }}>
              <label style={labelStyle}>Age: <span style={{ color:'#6366f1' }}>{form.age} years</span></label>
              <input type="range" min={5} max={17} value={form.age} onChange={e=>setForm({...form,age:+e.target.value})} style={{ width:'100%', accentColor:'#6366f1' }}/>
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:12, color:'#94a3b8', marginTop:4 }}><span>5 yrs</span><span>17 yrs</span></div>
            </div>
            <div style={{ marginBottom:16 }}>
              <label style={labelStyle}>Gender</label>
              <div style={{ display:'flex', gap:16 }}>
                {['Female','Male','Other'].map(g=>(
                  <label key={g} style={{ display:'flex', alignItems:'center', gap:6, cursor:'pointer', fontSize:14 }}>
                    <input type="radio" name="gender" value={g} checked={form.gender===g} onChange={()=>setForm({...form,gender:g})} style={{ accentColor:'#6366f1' }}/> {g}
                  </label>
                ))}
              </div>
            </div>
            <div style={{ marginBottom:16 }}>
              <label style={labelStyle}>Division</label>
              <select style={inputStyle} value={form.division} onChange={e=>setForm({...form,division:e.target.value})}>
                {DIVISIONS.map(d=><option key={d}>{d}</option>)}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Area Type</label>
              <div style={{ display:'flex', background:'#f1f5f9', borderRadius:8, padding:4, width:'fit-content', gap:4 }}>
                {['Urban','Rural'].map(a=>(
                  <button key={a} onClick={()=>setForm({...form,area:a})} style={{ padding:'6px 20px', borderRadius:6, border:'none', cursor:'pointer', fontSize:14, fontWeight:500, background: form.area===a?'#6366f1':'transparent', color: form.area===a?'white':'#64748b', transition:'all 0.2s' }}>{a}</button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Step 2 */}
        {step===1 && (
          <div>
            <h2 style={{ fontSize:16, fontWeight:600, marginBottom:20 }}>SDQ & CPSS Assessment</h2>
            {SDQ_ITEMS.map((item,i)=>(
              <div key={i} style={{ marginBottom:18 }}>
                <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                  <label style={{ fontSize:13, fontWeight:600 }}>{item.bn} <span style={{ color:'#94a3b8', fontWeight:400 }}>({item.en})</span></label>
                  <span style={{ fontWeight:700, color:'#6366f1', fontSize:14 }}>{form.sdq[i]}/10</span>
                </div>
                <input type="range" min={0} max={10} value={form.sdq[i]} onChange={e=>{ const s=[...form.sdq]; s[i]=+e.target.value; setForm({...form,sdq:s}); }} style={{ width:'100%', accentColor:'#6366f1' }}/>
              </div>
            ))}
            <div style={{ background:'#f8fafc', borderRadius:8, padding:14, marginBottom:18 }}>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
                <span style={{ fontSize:13, fontWeight:600 }}>SDQ Total Score</span>
                <span style={{ fontWeight:700, fontSize:18, color: sdqTotal>30?'#ef4444': sdqTotal>20?'#f97316':'#22c55e' }}>{sdqTotal}/50</span>
              </div>
              <div style={{ background:'#e2e8f0', borderRadius:99, height:8 }}>
                <div style={{ background:'#6366f1', width:`${(sdqTotal/50)*100}%`, height:8, borderRadius:99, transition:'width 0.3s' }}/>
              </div>
            </div>
            <div>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                <label style={{ fontSize:13, fontWeight:600 }}>CPSS Score (Child PTSD Symptom Scale)</label>
                <span style={{ fontWeight:700, color:'#6366f1' }}>{form.cpss}/85</span>
              </div>
              <input type="range" min={0} max={85} value={form.cpss} onChange={e=>setForm({...form,cpss:+e.target.value})} style={{ width:'100%', accentColor:'#6366f1' }}/>
            </div>
          </div>
        )}

        {/* Step 3 */}
        {step===2 && (
          <div>
            <h2 style={{ fontSize:16, fontWeight:600, marginBottom:8 }}>Narrative Description</h2>
            <p style={{ fontSize:13, color:'#64748b', marginBottom:16 }}>শিশুর পরিস্থিতি বিস্তারিত বর্ণনা করুন (বাংলায় লিখুন)</p>
            <textarea style={{ ...inputStyle, minHeight:160, resize:'vertical', lineHeight:1.6 }} value={form.narrative} onChange={e=>setForm({...form,narrative:e.target.value})} placeholder="এখানে বর্ণনা লিখুন..."/>
            <div style={{ display:'flex', justifyContent:'flex-end', fontSize:12, color:'#94a3b8', marginTop:4, marginBottom:16 }}>{form.narrative.length} characters</div>
            <div>
              <p style={{ fontSize:13, fontWeight:600, marginBottom:10, color:'#374151' }}>Sample Prompts:</p>
              {SAMPLE_PROMPTS.map((p,i)=>(
                <button key={i} onClick={()=>setForm({...form,narrative:p})} style={{ display:'block', width:'100%', textAlign:'left', padding:'8px 12px', background:'#f8fafc', border:'1px solid #e2e8f0', borderRadius:8, fontSize:13, color:'#374151', cursor:'pointer', marginBottom:6, transition:'all 0.2s', fontFamily:'Inter,sans-serif' }}
                  onMouseEnter={e=>e.currentTarget.style.background='#ede9fe'}
                  onMouseLeave={e=>e.currentTarget.style.background='#f8fafc'}
                >{p}</button>
              ))}
            </div>
          </div>
        )}

        {/* Step 4 */}
        {step===3 && (
          <div>
            <h2 style={{ fontSize:16, fontWeight:600, marginBottom:16 }}>Image Upload</h2>
            <div
              onClick={()=>fileRef.current.click()}
              onDrop={e=>{ e.preventDefault(); const files=[...e.dataTransfer.files]; setForm({...form,images:files.map(f=>URL.createObjectURL(f))}); }}
              onDragOver={e=>e.preventDefault()}
              style={{ border:'2px dashed #6366f1', borderRadius:12, padding:40, textAlign:'center', cursor:'pointer', background:'#fafafa', transition:'all 0.2s' }}
              onMouseEnter={e=>{ e.currentTarget.style.background='#ede9fe'; e.currentTarget.style.borderColor='#4f46e5'; }}
              onMouseLeave={e=>{ e.currentTarget.style.background='#fafafa'; e.currentTarget.style.borderColor='#6366f1'; }}
            >
              <Upload size={36} color="#6366f1" style={{ margin:'0 auto 12px' }}/>
              <p style={{ fontWeight:600, fontSize:14, color:'#374151' }}>Drag & drop images here</p>
              <p style={{ fontSize:13, color:'#94a3b8', marginTop:4 }}>or click to browse (JPG, PNG, max 5MB)</p>
              <input ref={fileRef} type="file" multiple accept="image/*" style={{ display:'none' }} onChange={e=>{ const files=[...e.target.files]; setForm({...form,images:files.map(f=>URL.createObjectURL(f))}); }}/>
            </div>
            {form.images.length>0 && (
              <div style={{ display:'flex', gap:12, flexWrap:'wrap', marginTop:16 }}>
                {form.images.map((src,i)=><img key={i} src={src} alt="" style={{ width:100, height:100, objectFit:'cover', borderRadius:8, border:'2px solid #6366f1' }}/>)}
              </div>
            )}
          </div>
        )}

        {/* Step 5 */}
        {step===4 && (
          <div style={{ textAlign:'center' }}>
            {loading ? (
              <div style={{ padding:40 }}>
                <div style={{ width:60, height:60, border:'4px solid #e2e8f0', borderTop:'4px solid #6366f1', borderRadius:'50%', margin:'0 auto 20px', animation:'spin 1s linear infinite' }}/>
                <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
                <p style={{ fontSize:16, fontWeight:600, color:'#374151' }}>Analyzing with AI...</p>
                <p style={{ fontSize:13, color:'#94a3b8', marginTop:8 }}>Processing SDQ, CPSS, narrative & image features</p>
              </div>
            ) : done ? (
              <div>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'center', marginBottom:20 }}>
                  <div style={{ position:'relative', width:120, height:120 }}>
                    <svg viewBox="0 0 120 120" style={{ transform:'rotate(-90deg)', width:120, height:120 }}>
                      <circle cx="60" cy="60" r="50" fill="none" stroke="#e2e8f0" strokeWidth="10"/>
                      <circle cx="60" cy="60" r="50" fill="none" stroke={riskColor} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${2*Math.PI*50}`} strokeDashoffset={`${2*Math.PI*50*(1-overallScore/100)}`}/>
                    </svg>
                    <div style={{ position:'absolute', inset:0, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center' }}>
                      <span style={{ fontSize:26, fontWeight:700, color: riskColor }}>{overallScore}%</span>
                    </div>
                  </div>
                </div>
                <div style={{ display:'inline-flex', alignItems:'center', gap:8, background: risk==='Critical'?'#fef2f2': risk==='High'?'#fff7ed': risk==='Moderate'?'#fefce8':'#f0fdf4', border:`1px solid ${riskColor}30`, borderRadius:20, padding:'6px 18px', marginBottom:20 }}>
                  <span style={{ width:10, height:10, borderRadius:'50%', background:riskColor }}/>
                  <span style={{ fontWeight:700, color:riskColor, fontSize:16 }}>{risk} Risk</span>
                </div>
                <div style={{ background:'#f8fafc', borderRadius:10, padding:16, marginBottom:16, textAlign:'left' }}>
                  <p style={{ fontWeight:600, marginBottom:10, fontSize:14 }}>Top 3 Risk Drivers</p>
                  {[['SDQ Score', sdqTotal, 50],['CPSS Score', form.cpss, 85],['Narrative Sentiment', Math.min(form.narrative.length/5,100), 100]].map(([label,val,max])=>(
                    <div key={label} style={{ marginBottom:10 }}>
                      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4, fontSize:13 }}><span>{label}</span><span style={{ fontWeight:600 }}>{Math.round((val/max)*100)}%</span></div>
                      <div style={{ background:'#e2e8f0', borderRadius:99, height:6 }}><div style={{ background:'#6366f1', width:`${Math.round((val/max)*100)}%`, height:6, borderRadius:99 }}/></div>
                    </div>
                  ))}
                </div>
                <div style={{ background:'#fef2f2', border:'1px solid #fecaca', borderRadius:10, padding:16, textAlign:'left' }}>
                  <p style={{ fontWeight:700, color:'#991b1b', marginBottom:8 }}>Referral Recommended</p>
                  <div style={{ display:'flex', gap:12 }}>
                    <a href="tel:1098" style={{ display:'flex', alignItems:'center', gap:6, background:'#ef4444', color:'white', borderRadius:8, padding:'8px 14px', textDecoration:'none', fontSize:13, fontWeight:600 }}><Phone size={14}/> DSS 1098</a>
                    <a href="tel:16767" style={{ display:'flex', alignItems:'center', gap:6, background:'#6366f1', color:'white', borderRadius:8, padding:'8px 14px', textDecoration:'none', fontSize:13, fontWeight:600 }}><Phone size={14}/> OCC 16767</a>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ padding:20 }}>
                <CheckCircle size={48} color="#22c55e" style={{ margin:'0 auto 16px' }}/>
                <p style={{ fontSize:16, fontWeight:600 }}>Ready to Submit</p>
                <p style={{ fontSize:13, color:'#64748b', marginTop:8 }}>Click Analyze to process this screening with AI.</p>
              </div>
            )}
          </div>
        )}

        {/* Navigation */}
        <div style={{ display:'flex', justifyContent:'space-between', marginTop:28, paddingTop:20, borderTop:'1px solid #e2e8f0' }}>
          <button onClick={()=>setStep(s=>Math.max(0,s-1))} disabled={step===0} style={{ padding:'9px 20px', borderRadius:8, border:'1px solid #e2e8f0', background:'white', fontSize:14, cursor: step===0?'not-allowed':'pointer', color: step===0?'#94a3b8':'#374151' }}>← Back</button>
          {step<4 && <button onClick={()=>setStep(s=>s+1)} style={{ padding:'9px 24px', borderRadius:8, border:'none', background:'#6366f1', color:'white', fontSize:14, fontWeight:600, cursor:'pointer' }}>Next →</button>}
          {step===4 && !done && <button onClick={handleSubmit} style={{ padding:'9px 24px', borderRadius:8, border:'none', background:'#6366f1', color:'white', fontSize:14, fontWeight:600, cursor:'pointer' }}>Analyze with AI</button>}
        </div>
      </div>
      <Footer/>
    </div>
  );
}
