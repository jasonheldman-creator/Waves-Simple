'use client';
import { useEffect, useRef } from 'react';

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const stars = Array.from({ length: 160 }, () => ({
      x: Math.random(), y: Math.random(),
      r: Math.random() * 1.3 + 0.2,
      o: Math.random() * 0.55 + 0.15,
      sp: Math.random() * 0.006 + 0.002,
      ph: Math.random() * Math.PI * 2,
    }));
    let frame = 0;
    let raf: number;
    function draw() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      stars.forEach(s => {
        const tw = Math.sin(frame * s.sp + s.ph);
        const alpha = Math.max(0.08, s.o + tw * 0.14);
        ctx.beginPath();
        ctx.arc(s.x * canvas.width, s.y * canvas.height, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(168,210,240,${alpha})`;
        ctx.fill();
      });
      frame++;
      raf = requestAnimationFrame(draw);
    }
    draw();
    const resize = () => {
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        body{background:#020912;color:#f0f4ff;font-family:'Inter',sans-serif;font-weight:300;min-height:100vh;overflow-x:hidden}
        .nebula{position:fixed;border-radius:50%;pointer-events:none;z-index:0}
        .n1{width:60vw;height:60vw;top:-15%;right:-10%;background:radial-gradient(circle,rgba(0,100,200,0.09) 0%,transparent 70%);animation:neb 14s ease-in-out infinite alternate}
        .n2{width:50vw;height:50vw;bottom:-20%;left:-8%;background:radial-gradient(circle,rgba(0,60,160,0.07) 0%,transparent 70%);animation:neb 18s ease-in-out infinite alternate-reverse}
        @keyframes neb{from{opacity:.7;transform:scale(1)}to{opacity:1;transform:scale(1.1)}}
        #vignette{position:fixed;inset:0;background:radial-gradient(ellipse at 50% 50%,transparent 40%,rgba(2,9,18,.20) 65%,rgba(2,9,18,.55) 85%,rgba(2,9,18,.80) 100%);pointer-events:none;z-index:9997}
        #kl{position:fixed;inset:0;pointer-events:none;z-index:9996}
        #kl::before{content:'';position:absolute;inset:0;background:linear-gradient(to right,rgba(0,120,200,.05) 0%,transparent 25%)}
        #kl::after{content:'';position:absolute;inset:0;background:linear-gradient(to left,rgba(0,200,255,.06) 0%,transparent 25%)}
        nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:18px 40px;display:flex;align-items:center;justify-content:space-between;background:rgba(2,9,18,.75);backdrop-filter:blur(20px);border-bottom:.5px solid rgba(77,184,255,.22)}
        .nl{font-family:'Inter',sans-serif;font-weight:500;font-size:15px;letter-spacing:.06em;color:#f0f4ff}
        .nl span{color:#00c8ff}