import { useState } from 'react';

function App(){
  const [file,setFile]=useState(null);
  const [modelA,setModelA]=useState('srcnn_x2');
  const [modelB,setModelB]=useState('espcn_x4');
  const [outA,setOutA]=useState('');
  const [outB,setOutB]=useState('');
  async function compare(){
    const fd=new FormData();
    fd.append('file',file);
    fd.append('model_a',modelA);
    fd.append('model_b',modelB);
    const res=await fetch('http://localhost:8000/api/v1/compare',{method:'POST',body:fd});
    const data=await res.json();
    setOutA('data:image/png;base64,'+data.images[modelA]);
    setOutB('data:image/png;base64,'+data.images[modelB]);
  }
  return (
    <div style={{padding:20,fontFamily:'sans-serif'}}>
      <h1>Super-Resolution Compare</h1>
      <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])}/>
      <div style={{marginTop:10}}>
        <label>Model A: </label>
        <select value={modelA} onChange={e=>setModelA(e.target.value)}>
          <option value="srcnn_x2">SRCNN x2</option>
          <option value="espcn_x4">ESPCN x4</option>
        </select>
        <label style={{marginLeft:10}}>Model B: </label>
        <select value={modelB} onChange={e=>setModelB(e.target.value)}>
          <option value="espcn_x4">ESPCN x4</option>
          <option value="srcnn_x2">SRCNN x2</option>
        </select>
      </div>
      <button onClick={compare} style={{marginTop:10}}>Compare</button>
      <div style={{display:'flex',gap:20,marginTop:20}}>
        {outA && <div><h3>{modelA}</h3><img src={outA} style={{maxWidth:'400px',border:'1px solid #ccc'}}/></div>}
        {outB && <div><h3>{modelB}</h3><img src={outB} style={{maxWidth:'400px',border:'1px solid #ccc'}}/></div>}
      </div>
    </div>
  );
}
export default App;
