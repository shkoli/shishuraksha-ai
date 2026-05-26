export const DIVISIONS = ['Dhaka','Chittagong','Rajshahi','Khulna','Barishal','Sylhet','Rangpur','Mymensingh'];

export const CASES = [
  { id:'SR-2024-001', age:8,  gender:'Female', division:'Dhaka',       risk:'Critical', score:89, confidence:91, date:'2024-11-01', sdq:18, cpss:72, narrative:85, image:60,
    screeningHistory:[
      { date:'2024-09-10', risk:'Moderate', score:0.65, confidence:78 },
      { date:'2024-10-05', risk:'High',     score:0.71, confidence:82 },
      { date:'2024-11-01', risk:'Critical', score:0.89, confidence:91 },
    ] },
  { id:'SR-2024-002', age:13, gender:'Male',   division:'Chittagong',  risk:'High',     score:74, confidence:82, date:'2024-11-03', sdq:14, cpss:58, narrative:70, image:45,
    screeningHistory:[
      { date:'2024-08-15', risk:'High',     score:0.78, confidence:80 },
      { date:'2024-09-12', risk:'High',     score:0.65, confidence:77 },
      { date:'2024-10-10', risk:'Moderate', score:0.52, confidence:74 },
      { date:'2024-11-03', risk:'Moderate', score:0.41, confidence:82 },
    ] },
  { id:'SR-2024-003', age:6,  gender:'Female', division:'Sylhet',      risk:'Critical', score:91, confidence:94, date:'2024-11-05', sdq:20, cpss:80, narrative:90, image:70,
    screeningHistory:[
      { date:'2024-09-01', risk:'Critical', score:0.88, confidence:90 },
      { date:'2024-10-01', risk:'Critical', score:0.91, confidence:93 },
      { date:'2024-11-05', risk:'Critical', score:0.87, confidence:94 },
    ] },
  { id:'SR-2024-004', age:15, gender:'Male',   division:'Rajshahi',    risk:'Moderate', score:52, confidence:73, date:'2024-11-06', sdq:10, cpss:40, narrative:55, image:30,
    screeningHistory:[
      { date:'2024-09-05', risk:'Moderate', score:0.48, confidence:70 },
      { date:'2024-10-03', risk:'Low',      score:0.39, confidence:72 },
      { date:'2024-11-06', risk:'Low',      score:0.28, confidence:73 },
    ] },
  { id:'SR-2024-005', age:11, gender:'Female', division:'Khulna',      risk:'High',     score:78, confidence:85, date:'2024-11-07', sdq:16, cpss:62, narrative:75, image:50,
    screeningHistory:[
      { date:'2024-09-08', risk:'Moderate', score:0.55, confidence:76 },
      { date:'2024-10-06', risk:'High',     score:0.67, confidence:80 },
      { date:'2024-11-07', risk:'High',     score:0.78, confidence:85 },
    ] },
  { id:'SR-2024-006', age:9,  gender:'Male',   division:'Barishal',    risk:'Moderate', score:48, confidence:69, date:'2024-11-08', sdq:9,  cpss:35, narrative:50, image:25 },
  { id:'SR-2024-007', age:14, gender:'Female', division:'Rangpur',     risk:'Low',      score:22, confidence:88, date:'2024-11-09', sdq:5,  cpss:15, narrative:20, image:10 },
  { id:'SR-2024-008', age:7,  gender:'Male',   division:'Mymensingh',  risk:'Critical', score:93, confidence:96, date:'2024-11-10', sdq:22, cpss:85, narrative:92, image:75 },
  { id:'SR-2024-009', age:16, gender:'Female', division:'Dhaka',       risk:'High',     score:71, confidence:79, date:'2024-11-11', sdq:13, cpss:55, narrative:68, image:42 },
  { id:'SR-2024-010', age:10, gender:'Male',   division:'Chittagong',  risk:'Low',      score:18, confidence:91, date:'2024-11-12', sdq:4,  cpss:12, narrative:15, image:8  },
  { id:'SR-2024-011', age:12, gender:'Female', division:'Sylhet',      risk:'Moderate', score:55, confidence:72, date:'2024-11-13', sdq:11, cpss:42, narrative:58, image:32 },
  { id:'SR-2024-012', age:8,  gender:'Male',   division:'Rajshahi',    risk:'High',     score:76, confidence:83, date:'2024-11-14', sdq:15, cpss:60, narrative:72, image:48 },
  { id:'SR-2024-013', age:13, gender:'Female', division:'Khulna',      risk:'Moderate', score:50, confidence:68, date:'2024-11-15', sdq:10, cpss:38, narrative:52, image:28 },
  { id:'SR-2024-014', age:11, gender:'Male',   division:'Rangpur',     risk:'Low',      score:25, confidence:86, date:'2024-11-16', sdq:6,  cpss:18, narrative:22, image:12 },
  { id:'SR-2024-015', age:9,  gender:'Female', division:'Barishal',    risk:'Moderate', score:58, confidence:75, date:'2024-11-17', sdq:12, cpss:45, narrative:60, image:35 },
];

export const RISK_COLORS = {
  Critical: { bg:'#fef2f2', text:'#991b1b', border:'#fecaca', dot:'#ef4444' },
  High:     { bg:'#fff7ed', text:'#9a3412', border:'#fed7aa', dot:'#f97316' },
  Moderate: { bg:'#fefce8', text:'#854d0e', border:'#fde68a', dot:'#eab308' },
  Low:      { bg:'#f0fdf4', text:'#166534', border:'#bbf7d0', dot:'#22c55e' },
};

export const RISK_DRIVERS = [
  { name:'SDQ Score', value:82 },
  { name:'CPSS Score', value:74 },
  { name:'Narrative Sentiment', value:68 },
  { name:'Image Features', value:55 },
  { name:'Age Factor', value:48 },
];

export const DIVISION_COUNTS = DIVISIONS.map(div => ({
  division: div,
  count: CASES.filter(c => c.division === div).length,
}));

export const SCREENING_TREND = Array.from({length:30}, (_, i) => ({
  day: `Nov ${i+1}`,
  count: Math.floor(Math.random()*8) + 1,
}));
