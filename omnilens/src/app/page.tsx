'use client';

import { useState, FormEvent, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getApiUrl } from '@/lib/config';
import { Search, Loader2, Aperture, AlertCircle, ShoppingBag, Terminal, Activity, Zap, CheckCircle2, MessageSquare, X, Send, Bot, Heart, ShoppingCart, SlidersHorizontal, Tag } from 'lucide-react';
import { useStore, type ProductItem } from '@/store/useStore';
import Link from 'next/link';
import WeightTuner from '@/components/WeightTuner';

export default function Home() {
  const [prompt, setPrompt] = useState('I want to go for a skiing trip, help me shop for the necessary items.');
  const [budgetStr, setBudgetStr] = useState('50000');
  const [noBudget, setNoBudget] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [queryType, setQueryType] = useState<string | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !hasStarted) {
        setHasStarted(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [hasStarted]);

  // ── Global store bindings ─────────────────────────────────────────
  const items = useStore((s) => s.items);
  const allItems = useStore((s) => s.allItems);
  const statusMessage = useStore((s) => s.statusMessage);
  const isProcessing = useStore((s) => s.isProcessing);
  const fallback = useStore((s) => s.fallback);
  const logs = useStore((s) => s.logs);
  const itemsHistory = useStore((s) => s.itemsHistory);
  const cartItems = useStore((s) => s.cartItems);
  const wishlist = useStore((s) => s.wishlist);
  const chatMessages = useStore((s) => s.chatMessages);

  const setStoreItems = useStore((s) => s.setItems);
  const setAllItems = useStore((s) => s.setAllItems);
  const setStatusMessage = useStore((s) => s.setStatusMessage);
  const setIsProcessing = useStore((s) => s.setIsProcessing);
  const setFallback = useStore((s) => s.setFallback);
  const addLog = useStore((s) => s.addLog);
  const clearLogs = useStore((s) => s.clearLogs);
  const pushHistory = useStore((s) => s.pushHistory);
  const popHistory = useStore((s) => s.popHistory);
  const resetItems = useStore((s) => s.resetItems);
  const addToCart = useStore((s) => s.addToCart);
  const toggleWishlist = useStore((s) => s.toggleWishlist);
  const addChatMessage = useStore((s) => s.addChatMessage);

  // ── Local UI state ────────────────────────────────────────────────
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [tunerOpen, setTunerOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  // ── Main search handler ───────────────────────────────────────────
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsProcessing(true);
    setStoreItems([]);
    setAllItems([]);
    setFallback(null);
    setQueryType(null);
    clearLogs();
    setStatusMessage('Uplinking to ML Engine: Parsing Intent Vector...');
    const sessionPrefix = `s${Date.now()}_`;

    try {
      const res = await fetch(`${getApiUrl()}/api/stream_shop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, budgetStr: noBudget ? '' : budgetStr }),
      });
      if (!res.body) throw new Error('No readable stream.');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (let part of parts) {
          part = part.trim();
          if (!part || !part.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(part.substring(6).trim());
            if (!payload.event || !payload.data) continue;
            const { event, data } = payload;

            if (event === 'expansion') {
              setStatusMessage(`Grid Extrapolated: Identified ${data.items.length} topological nodes`);
              if (data.intent_type) setQueryType(data.intent_type);
              const prefixedItems = data.items.map((it: ProductItem) => ({ ...it, id: sessionPrefix + it.id }));
              setStoreItems(prefixedItems);
              setAllItems(prefixedItems);
            } else if (event === 'pivot') {
              setFallback({ message: data.message, suggestedRentals: data.suggestedRentals });
            } else if (event === 'item_update') {
              const targetId = sessionPrefix + data.id;
              const items = useStore.getState().items;
              const exists = items.some(i => i.id === targetId);
              if (exists) {
                setStoreItems(
                  items.map((item) =>
                    item.id === targetId ? { ...item, status: data.status, statusText: data.statusText, progress: data.progress } : item
                  )
                );
              } else {
                // If it's a new sub-item (e.g. 1_0) starting analysis, add it as a placeholder
                setStoreItems([...items, { ...data, id: targetId, name: data.statusText || 'Processing...', status: 'analyzing', progress: data.progress, category: 'Components', external_link: '#' }]);
              }
            } else if (event === 'item_finish') {
              const targetId = sessionPrefix + data.id;
              const baseId = targetId.split('_')[0];
              const currentItems = useStore.getState().items;

              // Remove the generic placeholder (e.g. ID "1") if we're receiving a specific result for it
              const filtered = currentItems.filter(it => it.id !== baseId);
              const exists = filtered.some(i => i.id === targetId);

              let updated;
              if (exists) {
                updated = filtered.map((item) => (item.id === targetId ? { ...item, ...data, id: targetId } : item));
              } else {
                updated = [...filtered, { ...data, id: targetId }];
              }
              setStoreItems(updated);
              setAllItems(updated);
            } else if (event === 'log') {
              addLog(data.message);
            } else if (event === 'done') {
              setStatusMessage('Pipeline sequence complete.');
              setIsProcessing(false);
              // Clean up any remaining "pending" placeholders from the expansion
              const finalItems = useStore.getState().items.filter(it => it.status === 'complete');
              if (finalItems.length > 0) {
                setStoreItems(finalItems);
                setAllItems(finalItems);
              }
            } else if (event === 'error') {
              setStatusMessage(`Critical failure: ${data.message}`);
              setIsProcessing(false);
            }
          } catch (err) {
            console.warn('Failed to parse SSE chunk', err);
          }
        }
      }
    } catch (error) {
      console.error(error);
      setIsProcessing(false);
      setStatusMessage('Connection severed. Reboot required.');
    }
  };

  // ── Chat handler ──────────────────────────────────────────────────
  const sendChatMessage = async () => {
    const msg = chatInput.trim();
    if (!msg || isChatLoading) return;
    setChatInput('');
    setIsChatLoading(true);
    addChatMessage({ id: Date.now().toString(), role: 'user', content: msg });
    const ml = msg.toLowerCase();

    if (['undo', 'rollback', 'go back', 'revert'].some((t) => ml.includes(t))) {
      if (itemsHistory.length > 0) { popHistory(); addChatMessage({ id: (Date.now() + 1).toString(), role: 'ai', content: '↩️ Rolled back to your previous item list.' }); }
      else addChatMessage({ id: (Date.now() + 1).toString(), role: 'ai', content: "⚠️ Nothing to undo — you're at the original state." });
      setIsChatLoading(false); return;
    }
    if (['reset', 'start over', 'restore'].some((t) => ml.includes(t))) {
      resetItems(); addChatMessage({ id: (Date.now() + 1).toString(), role: 'ai', content: '🔄 Reset complete! All items restored.' });
      setIsChatLoading(false); return;
    }
    const rankMatch = ml.match(/(?:show|top|list|rank|find)\s+(?:me\s+)?(?:top\s+)?(\d+)?\s*(.+)/);
    const isRankCmd = ['show me top', 'list top', 'rank top', 'top 10', 'top 5'].some((t) => ml.includes(t));
    if (isRankCmd && rankMatch) {
      // Logic for local ranking is fine, but if user explicitly asks for 'top 10', 
      // we often want a fresh backend Deep Search. 
      // I'll keep local for quick filters but fall through if they ask for something new.
      const pool = allItems.length > 0 ? allItems : items;
      const query = rankMatch[2]?.trim().toLowerCase() || '';
      const localMatches = pool.filter(i => i.name.toLowerCase().includes(query));
      if (localMatches.length >= 5) { // If we have enough local data, show it
        const count = rankMatch[1] ? parseInt(rankMatch[1]) : 10;
        const matched = localMatches.sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, count);
        const lines = matched.map((item, i) => `${i + 1}. ${item.name} — ₹${(item.finalPrice || 0).toLocaleString()} | Score: ${(item.score || 0).toFixed(1)}/100`).join('\n');
        addChatMessage({ id: (Date.now() + 1).toString(), role: 'ai', content: `🏆 Top ${matched.length} (Local) for "${query}":\n\n${lines}` });
        setIsChatLoading(false); return;
      }
    }
    if (['end shopping', 'checkout', 'done shopping', 'finish', 'end session'].some((t) => ml.includes(t))) {
      setChatOpen(false);
      addChatMessage({ id: (Date.now() + 1).toString(), role: 'ai', content: '🛒 Head to your Cart page to review and confirm your order!' });
      setIsChatLoading(false); return;
    }

    try {
      const res = await fetch(`${getApiUrl()}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, items: items.map((i) => ({ id: i.id, name: i.name, finalPrice: i.finalPrice })), wishlist }),
      });
      const data = await res.json();
      const currentItems = useStore.getState().items;
      if (data.action === 'remove' && data.remove_id) {
        pushHistory(); setStoreItems(currentItems.filter((i) => i.id !== data.remove_id));
      }
      else if (data.action === 'replace') {
        pushHistory(); const ni: ProductItem = { ...data.new_item, status: 'complete', progress: 100, category: 'Components' };
        if (data.remove_id) setStoreItems(currentItems.map((i) => i.id === data.remove_id ? ni : i)); else setStoreItems([...currentItems, ni]);
      }
      else if (data.action === 'add' && data.new_item) {
        pushHistory(); setStoreItems([...currentItems, { ...data.new_item, status: 'complete', progress: 100, category: 'Components' }]);
      }
      else if (data.action === 'add_bulk' && data.items) {
        pushHistory();
        const bulkItems = data.items.map((it: any) => ({ ...it, status: 'complete', progress: 100, category: 'Components' }));
        setStoreItems([...currentItems, ...bulkItems]);
      }
      addChatMessage({ id: Date.now().toString(), role: 'ai', content: data.message || '✅ *Instruction executed.*' });
    } catch {
      addChatMessage({ id: Date.now().toString(), role: 'ai', content: '❌ *Critical: Uplink failed.* Ensure ML Engine is active.' });
    } finally { setIsChatLoading(false); }
  };

  const handleExploreFurther = async () => {
    if (isProcessing) return;
    setIsProcessing(true);
    setStatusMessage('Extrapolating deeper market dimensions...');
    try {
      const res = await fetch(`${getApiUrl()}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: '[SYSTEM] EXTRAPOLATE_MORE: ' + prompt, items: items.map((i) => ({ id: i.id, name: i.name, target_query: i.target_query, finalPrice: i.finalPrice })) }),
      });
      const data = await res.json();
      if (data.action === 'add_bulk' && data.items) {
        pushHistory();
        const bulkItems = data.items.map((it: any) => ({ ...it, status: 'complete', progress: 100, category: 'Components' }));
        setStoreItems([...items, ...bulkItems]);
      }
    } catch {
      setStatusMessage('⚠️ Extrapolation failed.');
    } finally {
      setIsProcessing(false);
      setStatusMessage('System Online');
    }
  };

  const sendChatMessageInternal = async (msg: string) => {
    setIsChatLoading(true);
    addChatMessage({ id: Date.now().toString(), role: 'ai', content: `🧠 *Generative Sub-routine initiated.* Extrapolating fresh market dimensions for your query...` });

    try {
      const res = await fetch(`${getApiUrl()}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, items: items.map((i) => ({ id: i.id, name: i.name, finalPrice: i.finalPrice })), wishlist }),
      });
      const data = await res.json();
      const cur = useStore.getState().items;
      if (data.action === 'remove' && data.remove_id) { pushHistory(); setStoreItems(cur.filter((i) => i.id !== data.remove_id)); }
      else if (data.action === 'replace') { pushHistory(); const ni: ProductItem = { ...data.new_item, status: 'complete', progress: 100, category: 'Components' }; if (data.remove_id) setStoreItems(cur.map((i) => i.id === data.remove_id ? ni : i)); else setStoreItems([...cur, ni]); }
      else if (data.action === 'add' && data.new_item) { pushHistory(); setStoreItems([...cur, { ...data.new_item, status: 'complete', progress: 100, category: 'Components' }]); }
      else if (data.action === 'add_bulk' && data.items) { pushHistory(); const withDefaults = data.items.map((it: any) => ({ ...it, status: 'complete', progress: 100, category: 'Components' })); setStoreItems([...cur, ...withDefaults]); }
      addChatMessage({ id: Date.now().toString(), role: 'ai', content: data.message || '✅ *Instruction executed.*' });
    } catch {
      addChatMessage({ id: Date.now().toString(), role: 'ai', content: '⚠️ Uplink failed.' });
    } finally { setIsChatLoading(false); }
  };

  const renderProductCard = (item: ProductItem, idx: number) => (
    <motion.div key={item.id} initial={{ opacity: 0, y: 30, scale: 0.9 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.5, delay: idx * 0.05 }}
      className="break-inside-avoid relative group perspective-1000">
      <div className="absolute -inset-0.5 bg-gradient-to-br from-purple-500/20 to-cyan-500/0 rounded-2xl blur opacity-0 group-hover:opacity-100 transition duration-500" />
      <motion.div
        whileHover={{ rotateY: idx % 2 === 0 ? 5 : -5, rotateX: 2, scale: 1.02 }}
        className={`relative bg-card/60 backdrop-blur-xl border border-card-border rounded-2xl overflow-hidden shadow-2xl flex flex-col h-full transform transition-all duration-300 hover:border-slate-500/50 ${item.score && item.score > 90 ? 'holographic-card shadow-[0_0_30px_rgba(168,85,247,0.15)]' : ''}`}>

        {/* Card Image */}
        {item.image ? (
          <div className="relative h-48 w-full bg-white overflow-hidden p-4 group-hover:bg-slate-50 transition-colors">
            <div className="absolute inset-0 bg-gradient-to-t from-slate-900 to-transparent z-10" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={item.image} alt={item.name} className="w-full h-full object-contain filter contrast-125 mix-blend-multiply transition-transform duration-700 group-hover:scale-110" />

            <div className="absolute top-3 left-3 z-20 flex gap-2">
              <button onClick={() => toggleWishlist(item)}
                className={`w-8 h-8 rounded-full border flex items-center justify-center backdrop-blur-md transition-all shadow-lg ${wishlist.some((w) => w.id === item.id) ? 'bg-rose-500/80 border-rose-400 text-white shadow-[0_0_12px_rgba(244,63,94,0.5)]' : 'bg-black/40 border-white/20 text-slate-300 hover:bg-rose-500/30 hover:border-rose-400 hover:text-rose-300'}`}>
                <Heart className={`w-3.5 h-3.5 ${wishlist.some((w) => w.id === item.id) ? 'fill-current' : ''}`} />
              </button>
              <button onClick={() => addToCart(item)}
                className={`w-8 h-8 rounded-full border flex items-center justify-center backdrop-blur-md transition-all shadow-lg ${cartItems.find((c) => c.id === item.id) ? 'bg-emerald-500/80 border-emerald-400 text-white shadow-[0_0_12px_rgba(52,211,153,0.5)]' : 'bg-black/40 border-white/20 text-slate-300 hover:bg-emerald-500/30 hover:border-emerald-400 hover:text-emerald-300'}`}>
                <ShoppingCart className="w-3.5 h-3.5" />
              </button>
            </div>

            {item.coupon_applied && (
              <div className="absolute top-3 right-3 z-20">
                <div className="bg-rose-500 text-white text-[9px] font-black px-2 py-1 rounded shadow-lg flex items-center gap-1 border border-white/20">
                  <Tag className="w-3 h-3" /> {item.coupon_applied}
                </div>
              </div>
            )}

            <div className="absolute bottom-3 right-3 z-20 flex flex-col gap-1 items-end">
              <div className={`text-[9px] font-black px-2 py-0.5 rounded-full shadow-lg backdrop-blur-md border border-white/10 text-white bg-gradient-to-r from-purple-600 to-indigo-600`}>
                {item.sentiment ? `Reliability: ${item.sentiment}%` : 'Authentic'}
              </div>
              <div className={`text-[9px] font-bold px-2 py-0.5 rounded-full shadow-lg backdrop-blur-md border ${item.platform?.includes('Amazon') ? 'bg-orange-500/20 border-orange-500/50 text-orange-300' : 'bg-blue-500/20 border-blue-500/50 text-blue-300'}`}>{item.platform}</div>
            </div>
          </div>
        ) : (
          <div className="relative h-48 w-full overflow-hidden border-b border-white/5 flex items-center justify-center"
            style={{ background: `linear-gradient(135deg, hsl(${Math.abs((item.name?.charCodeAt(0) ?? 65) * 7) % 360}, 35%, 10%) 0%, hsl(${Math.abs((item.name?.charCodeAt(0) ?? 65) * 7 + 140) % 360}, 30%, 7%) 100%)` }}>
            <div className="flex flex-col items-center gap-2 opacity-60">
              <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center backdrop-blur-sm">
                <span className="text-xl font-black text-white/40">
                  {(item.name || 'NA').split(' ').slice(0, 2).map((w: string) => w[0] ?? '').join('').toUpperCase()}
                </span>
              </div>
              <span className="text-[9px] font-mono text-white/25 tracking-widest uppercase">Loading Image...</span>
            </div>
          </div>
        )}

        {/* Card Content */}
        <div className="p-5 flex-grow flex flex-col relative z-20 -mt-8 pt-8 bg-card/90 backdrop-blur-md rounded-t-2xl border-t border-card-border">
          <div className="flex justify-between items-start mb-2">
            <h3 className="text-sm font-bold text-foreground leading-tight group-hover:text-purple-300 transition-colors line-clamp-2">{item.name}</h3>
          </div>
          
          {item.reddit_sentiment && (
            <div className="flex flex-col gap-1 mb-3 bg-black/40 border border-white/5 rounded-lg p-2">
              <div className="flex items-center gap-1.5 line-clamp-2">
                <div className="w-4 h-4 rounded-full bg-orange-600 flex items-center justify-center shrink-0"><span className="text-[8px] text-white font-bold">r/</span></div>
                <span className="text-[10px] text-slate-300 italic">"{item.reddit_sentiment}"</span>
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 mb-4">
            <span className="text-[10px] text-slate-500 font-mono tracking-widest bg-card px-2 py-0.5 rounded-md border border-card-border whitespace-nowrap">{item.category || 'General'}</span>
            {item.tags?.map(tag => (
              <span key={tag} className="text-[9px] font-bold px-2 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300">{tag}</span>
            ))}
          </div>

          <div className="mt-auto">
            <div className="flex items-end justify-between">
              <div className="flex flex-col">
                <span className="text-[10px] text-slate-500 uppercase tracking-widest font-mono">Market Price</span>
                <div className="flex items-center gap-2">
                  {(item.finalPrice || 0) > 0 ? (
                    <span className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                      ₹{(item.finalPrice || 0).toLocaleString()}
                    </span>
                  ) : (
                    <span className="text-sm font-bold text-amber-500/70 border border-amber-500/20 px-2 py-0.5 rounded-md bg-amber-500/5">
                      Price N/A
                    </span>
                  )}
                  {item.wait_to_buy !== undefined && (
                    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${item.wait_to_buy ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}`}>
                      {item.wait_to_buy ? 'Wait' : 'Buy Now'}
                    </span>
                  )}
                </div>
              </div>
              <a href={item.external_link} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 bg-white text-black px-3 py-1.5 rounded-lg font-bold text-[10px] shadow-[0_0_15px_rgba(255,255,255,0.2)] hover:bg-slate-200 hover:scale-105 transition-all">
                <ShoppingBag className="w-3 h-3" /> BUY
              </a>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );

  return (
    <div className="min-h-screen text-foreground font-sans selection:bg-purple-500/30 overflow-hidden relative">
      <AnimatePresence mode="wait">
        {/* ── STUNNING INTRO SCREEN ────────────────────────────────── */}
        {!hasStarted && (
          <motion.div key="intro" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 1.1, filter: 'blur(20px)' }} transition={{ duration: 0.8, ease: "easeInOut" }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-[#02040a] overflow-hidden">
            
            {/* Immersive Background Effects */}
            <div className="absolute inset-0 pointer-events-none">
               <motion.div animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.4, 0.2] }} transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
                 className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[90vw] h-[90vw] max-w-[1400px] max-h-[1400px] rounded-full bg-[radial-gradient(circle,rgba(124,58,237,0.15)_0%,rgba(6,182,212,0.1)_40%,transparent_70%)] blur-[120px]" />
               
               <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:60px_60px] [mask-image:radial-gradient(ellipse_70%_70%_at_50%_50%,#000_30%,transparent_100%)]" />
               
               {/* Ascending Data Streams */}
               {[...Array(20)].map((_, i) => (
                 <motion.div key={`stream-${i}`} initial={{ y: "110vh", x: ((i * 17) % 100) + "vw", opacity: 0 }}
                   animate={{ y: "-10vh", opacity: [0, 1, 0] }} transition={{ duration: (i % 8) + 8, repeat: Infinity, delay: (i % 5), ease: "linear" }}
                   className="absolute w-px h-24 bg-gradient-to-t from-transparent via-cyan-500/50 to-transparent blur-[1px]" />
               ))}
            </div>

            {/* Central Glass Console */}
            <motion.div initial={{ scale: 0.9, opacity: 0, y: 40 }} animate={{ scale: 1, opacity: 1, y: 0 }} transition={{ delay: 0.3, type: 'spring', stiffness: 80, damping: 20 }}
              className="relative z-10 flex flex-col items-center justify-center p-10 md:p-20 rounded-[3rem] bg-white/[0.02] border border-white/[0.05] shadow-[0_0_100px_rgba(0,0,0,0.6)] backdrop-blur-3xl max-w-[90vw] w-auto mx-auto text-center cursor-pointer hover:bg-white/[0.04] transition-colors duration-500"
              onClick={() => setHasStarted(true)}>
              
              {/* Spinning Logo Prism */}
              <div className="relative mb-8">
                <div className="absolute inset-0 bg-gradient-to-tr from-purple-600 via-pink-500 to-cyan-400 rounded-full blur-[40px] opacity-30 animate-pulse" />
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 24, repeat: Infinity, ease: "linear" }}
                  className="w-24 h-24 md:w-32 md:h-32 relative z-10 flex items-center justify-center rounded-full bg-black/50 border border-white/10 backdrop-blur-xl shadow-2xl">
                  <div className="absolute inset-2 rounded-full border border-dashed border-white/20 animate-[spin_30s_linear_infinite_reverse]" />
                  <div className="absolute inset-5 rounded-full border top-0 bottom-0 left-0 right-0 border-cyan-500/20" />
                  <Aperture className="w-12 h-12 md:w-16 md:h-16 text-cyan-400 drop-shadow-[0_0_20px_rgba(34,211,238,0.9)]" />
                </motion.div>
              </div>

              {/* Title & Tagline */}
              <div className="space-y-4 mb-10 relative w-full px-6">
                <h1 className="text-4xl md:text-5xl lg:text-7xl font-light tracking-widest text-slate-300">
                  OMNILENS <span className="text-transparent bg-clip-text bg-gradient-to-br from-purple-400 to-cyan-400 font-bold">PRO</span>
                </h1>
                <p className="text-sm md:text-lg text-slate-500 font-mono tracking-widest uppercase animate-pulse">
                  Awaiting prompt sequence...
                </p>
              </div>
              
              <p className="text-slate-600 text-xs md:text-sm font-mono opacity-50 tracking-widest">PRESS SPACE OR CLICK TO INITIALIZE</p>
            </motion.div>
          </motion.div>
        )}

        {/* ── MAIN DASHBOARD ───────────────────────────────────────── */}
        {hasStarted && (
          <motion.div key="dashboard" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}
            className="flex flex-col xl:flex-row min-h-screen">

            {/* LEFT PANEL */}
            <div className="relative w-full xl:w-[400px] flex-shrink-0 xl:h-screen flex flex-col cyber-glass border-r border-card-border overflow-hidden data-stream">

              {/* Header */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-card-border bg-black/5 dark:bg-black/20 flex-shrink-0">
                <div className="flex items-center gap-3">
                  <div className="relative flex h-3 w-3"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" /><span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500" /></div>
                  <span className="font-mono text-xs tracking-widest text-slate-400 uppercase">System Status: <span className="text-cyan-400">Online</span></span>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => setChatOpen((o) => !o)} title="AI Assistant"
                    className={`relative p-1.5 rounded-lg transition-all duration-200 ${chatOpen ? 'bg-purple-500/30 text-purple-300 shadow-[0_0_12px_rgba(168,85,247,0.5)]' : 'text-slate-500 hover:text-purple-300 hover:bg-purple-500/20'}`}>
                    <MessageSquare className="w-5 h-5" />
                    {chatMessages.length > 1 && <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-purple-500 animate-pulse" />}
                  </button>
                  <Link href="/cart" className="relative p-1.5 rounded-lg text-slate-500 hover:text-emerald-300 hover:bg-emerald-500/20 transition-all">
                    <ShoppingCart className="w-5 h-5" />
                    {cartItems.length > 0 && <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 text-slate-900 text-[9px] font-black flex items-center justify-center">{cartItems.length}</span>}
                  </Link>
                  <button onClick={() => setTunerOpen((o) => !o)} title="Weight Tuner"
                    className={`relative p-1.5 rounded-lg transition-all duration-200 ${tunerOpen ? 'bg-cyan-500/30 text-cyan-300 shadow-[0_0_12px_rgba(6,182,212,0.5)]' : 'text-slate-500 hover:text-cyan-300 hover:bg-cyan-500/20'}`}>
                    <SlidersHorizontal className="w-5 h-5" />
                  </button>
                  <Aperture className="w-6 h-6 text-purple-500/80 drop-shadow-[0_0_8px_rgba(168,85,247,0.8)]" />
                </div>
              </div>

              {/* Command Input Area */}
              <form onSubmit={handleSubmit} className="px-5 pt-5 pb-3 flex-shrink-0">
                <div className="flex justify-between items-end mb-2">
                  <label className="block text-[10px] font-mono tracking-widest text-slate-500 uppercase">Mission Prompt</label>
                </div>
                <div className="relative">
                  <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} disabled={isProcessing}
                    className="w-full bg-white/[0.04] border border-white/[0.1] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-purple-500/50 resize-none transition-colors leading-relaxed disabled:opacity-60"
                    placeholder="Describe your shopping mission..." />
                </div>
                <div className="mt-3 space-y-2">
                  <label className="block text-[10px] font-mono tracking-widest text-slate-500 uppercase">Budget Constraint</label>
                  <div className="flex items-center gap-3">
                    <div className={`flex-grow flex items-center bg-white/[0.04] border border-white/[0.1] rounded-xl px-4 py-2.5 transition-colors focus-within:border-purple-500/50 ${noBudget ? 'opacity-40 cursor-not-allowed' : ''}`}>
                      <span className="text-slate-400 font-bold select-none mr-2">{noBudget ? '' : '₹'}</span>
                      <input type="text" value={noBudget ? 'Unlimited' : budgetStr} onChange={(e) => setBudgetStr(e.target.value.replace(/^₹\s*/, ''))}
                        disabled={noBudget || isProcessing}
                        className="bg-transparent w-full text-sm text-slate-200 placeholder-slate-600 outline-none disabled:cursor-not-allowed" />
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer flex-shrink-0">
                      <div className={`w-9 h-5 rounded-full transition-colors duration-300 relative ${noBudget ? 'bg-purple-500' : 'bg-slate-700'}`} onClick={() => setNoBudget((v) => !v)}>
                        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-300 ${noBudget ? 'translate-x-4' : 'translate-x-0.5'}`} />
                      </div>
                      <span className="text-[10px] font-mono text-slate-500">∞</span>
                    </label>
                  </div>
                </div>
                <button type="submit" disabled={isProcessing || !prompt.trim()}
                  className="mt-4 w-full group relative flex items-center justify-center gap-2 py-3 px-6 rounded-xl text-sm font-bold tracking-widest uppercase overflow-hidden transition-all duration-300
                  bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500
                  shadow-[0_0_20px_rgba(168,85,247,0.3)] hover:shadow-[0_0_30px_rgba(168,85,247,0.5)] disabled:shadow-none">
                  {isProcessing ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing...</> : <><Zap className="w-4 h-4" /> Initialize Matrix</>}
                </button>
              </form>

              {/* Live Engine Feed */}
              <div className="flex-grow relative min-h-0 px-5 pb-5">
                <div className="flex items-center gap-2 mb-3 mt-1">
                  <Terminal className="w-3.5 h-3.5 text-slate-600" />
                  <span className="text-[10px] font-mono tracking-widest text-slate-600 uppercase">Live Engine Feed</span>
                  {isProcessing && <span className="ml-auto flex h-1.5 w-1.5"><span className="animate-ping absolute inline-flex h-1.5 w-1.5 rounded-full bg-cyan-400 opacity-75" /><span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-cyan-500" /></span>}
                </div>
                <div className="h-full max-h-[180px] overflow-y-auto space-y-1.5 scrollbar-thin scrollbar-thumb-slate-800">
                  {logs.length === 0 && <p className="text-[11px] text-slate-700 font-mono italic">No activity yet...</p>}
                  <AnimatePresence>
                    {logs.map((log) => (
                      <motion.div key={log.id} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="text-[11px] font-mono break-words border-l-2 border-slate-800 pl-2">
                        <span className="text-slate-600 mr-2">[{log.time}]</span>
                        <span className={log.msg.includes('✅') ? 'text-emerald-400' : log.msg.includes('❌') || log.msg.includes('🛑') ? 'text-rose-400' : log.msg.includes('🔎') ? 'text-cyan-400' : 'text-slate-300'}>{log.msg}</span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  <div ref={logsEndRef} />
                </div>
                <div className="absolute bottom-0 left-0 w-full h-12 bg-gradient-to-t from-black via-black/80 to-transparent pointer-events-none" />
              </div>

              {/* ── CHAT PANEL ─────────────────────────────────────── */}
              <AnimatePresence>
                {chatOpen && (
                  <motion.div key="chat-panel" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}
                    className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-md p-4 sm:p-8 md:p-16">
                    <motion.div initial={{ scale: 0.95, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.95, y: 20 }} transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      className="w-full max-w-5xl h-full max-h-[85vh] flex flex-col bg-slate-50 dark:bg-slate-950/95 shadow-[0_0_100px_rgba(0,0,0,0.5)] border border-slate-200 dark:border-white/10 rounded-[2rem] overflow-hidden relative">
                    <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-[2rem] z-[-1]">
                      <div className="absolute top-[-20%] left-[-20%] w-[60%] h-[60%] rounded-full bg-purple-600/10 dark:bg-purple-900/40 blur-[100px]" />
                      <div className="absolute bottom-[20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-cyan-600/10 dark:bg-cyan-900/30 blur-[100px]" />
                    </div>
                    
                    {/* Chat Header */}
                    <div className="relative z-10 flex items-center justify-between px-6 py-5 border-b border-slate-200 dark:border-white/5 bg-white/50 dark:bg-black/20">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-cyan-500 p-[1px] shadow-lg shadow-purple-500/20">
                          <div className="w-full h-full bg-slate-50 dark:bg-slate-950 rounded-[11px] flex items-center justify-center">
                            <Bot className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                          </div>
                        </div>
                        <div>
                          <h3 className="text-sm font-black text-slate-800 dark:text-white leading-tight">OmniLens Core</h3>
                          <p className="text-[9px] text-emerald-600 dark:text-emerald-400 font-mono tracking-widest uppercase flex items-center gap-1.5 mt-0.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]" /> Active Session
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Link href="/chat" onClick={() => setChatOpen(false)} className="text-[10px] font-bold text-slate-500 dark:text-slate-400 hover:text-purple-600 dark:hover:text-purple-300 border border-slate-200 dark:border-white/10 bg-white dark:bg-white/5 hover:border-purple-300 dark:hover:border-purple-500/50 px-3 py-1.5 rounded-lg transition-all shadow-sm">Expand</Link>
                        <button onClick={() => setChatOpen(false)} className="w-8 h-8 rounded-full flex items-center justify-center bg-slate-200 dark:bg-white/5 text-slate-500 hover:bg-rose-500 hover:text-white dark:text-slate-400 dark:hover:bg-rose-500 transition-colors">
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    {/* Chat Messages */}
                    <div className="relative z-10 flex-grow overflow-y-auto px-6 py-6 space-y-5 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-800 pb-4">
                      {chatMessages.map((msg) => (
                        <motion.div key={msg.id} initial={{ opacity: 0, y: 10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          {msg.role === 'ai' && (
                            <div className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/50 border border-purple-200 dark:border-purple-500/30 flex items-center justify-center mr-3 shrink-0 self-end mb-1">
                              <Bot className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
                            </div>
                          )}
                          <div className={`max-w-[75%] rounded-2xl px-4 py-3 text-[13px] leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-gradient-to-br from-purple-600 to-cyan-600 text-white rounded-br-sm' : 'bg-white dark:bg-white/[0.04] border border-slate-200 dark:border-white/[0.08] text-slate-700 dark:text-slate-300 rounded-bl-sm'}`} style={{ whiteSpace: 'pre-line' }}>
                            {msg.content}
                          </div>
                        </motion.div>
                      ))}
                      {isChatLoading && (
                        <div className="flex justify-start items-end">
                          <div className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/50 border border-purple-200 dark:border-purple-500/30 flex items-center justify-center mr-3 shrink-0 mb-1">
                             <Bot className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
                          </div>
                          <div className="bg-white dark:bg-white/[0.04] border border-slate-200 dark:border-white/[0.08] rounded-2xl rounded-bl-sm px-5 py-3.5 flex gap-1.5 shadow-sm">
                            {[0, 1, 2].map((i) => (<motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-purple-500 dark:bg-purple-400" animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }} transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.15 }} />))}
                          </div>
                        </div>
                      )}
                      <div ref={chatEndRef} />
                    </div>
                    
                    {/* Chat Input */}
                    <div className="relative z-10 p-5 pt-3 border-t border-slate-200 dark:border-white/5 bg-slate-100/50 dark:bg-black/20">
                      <div className="flex gap-3 items-end bg-white dark:bg-slate-900 border border-slate-300 dark:border-white/10 rounded-2xl p-2 shadow-sm focus-within:border-purple-500/50 focus-within:ring-2 focus-within:ring-purple-500/20 transition-all">
                        <textarea value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); } }}
                          placeholder="Ask: 'total?', 'best pick?', 'compare', 'remove X'..." rows={1} disabled={isChatLoading}
                          className="flex-grow bg-transparent px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-600 outline-none resize-none max-h-32 min-h-[40px] scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-slate-800"
                          style={{ minHeight: '40px' }} />
                        <button onClick={sendChatMessage} disabled={isChatLoading || !chatInput.trim()}
                          className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center bg-slate-900 dark:bg-gradient-to-br dark:from-purple-600 dark:to-cyan-600 hover:bg-slate-800 dark:hover:from-purple-500 dark:hover:to-cyan-500 disabled:opacity-30 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white shadow-md transition-all">
                          <Send className="w-4 h-4 ml-0.5" />
                        </button>
                      </div>
                      {/* Quick-reply chips */}
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {["What's my total?", "Best pick?", "Compare items", "Undo"].map((chip) => (
                          <button key={chip} onClick={() => setChatInput(chip)}
                            className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-slate-100 dark:bg-white/5 border border-slate-200 dark:border-white/10 text-slate-500 dark:text-slate-400 hover:bg-purple-50 dark:hover:bg-purple-500/10 hover:border-purple-300 dark:hover:border-purple-500/30 hover:text-purple-600 dark:hover:text-purple-300 transition-all">
                            {chip}
                          </button>
                        ))}
                      </div>
                    </div>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* RIGHT PANEL: DATAGRID */}
            <div className="flex-grow relative z-0 h-[50vh] xl:h-screen overflow-y-auto p-6 xl:p-10 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">

              {/* Engine Status Header */}
              <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/5 pb-6">
                <div className="flex flex-col gap-2">
                  <h2 className="text-2xl font-light tracking-wide text-white flex items-center gap-3"><Activity className="w-5 h-5 text-purple-400" />{statusMessage}</h2>
                  {queryType && (
                    <span className={`w-max font-mono text-[10px] px-3 py-1 rounded-md border tracking-widest shadow-sm uppercase font-bold ${queryType === 'Definite Query' ? 'bg-blue-500/10 text-blue-400 border-blue-500/30' : 'bg-purple-500/10 text-purple-400 border-purple-500/30'}`}>
                      {queryType}
                    </span>
                  )}
                </div>
                {items.length > 0 && (
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-4 bg-card/50 backdrop-blur-md border border-card-border rounded-full px-5 py-2 shadow-inner">
                      <div className="flex flex-col"><span className="text-[10px] uppercase font-mono tracking-widest text-slate-500 leading-tight">Total Value</span><span className="text-base font-bold text-emerald-400 leading-tight">₹{items.reduce((acc, it) => acc + (it.finalPrice || 0), 0).toLocaleString()}</span></div>
                      <div className="w-px h-6 bg-slate-700" />
                      <div className="flex flex-col"><span className="text-[10px] uppercase font-mono tracking-widest text-slate-500 leading-tight">Objects</span><span className="text-base font-bold text-cyan-400 leading-tight">{items.filter((i) => i.status === 'complete').length} / {items.length}</span></div>
                    </div>
                    <Link href="/cart" className={`relative p-2.5 rounded-full border transition-all bg-card/50 border-card-border text-slate-400 hover:text-emerald-300 hover:border-emerald-500/50`}>
                      <ShoppingCart className="w-5 h-5" />
                      {cartItems.length > 0 && <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-emerald-500 text-slate-900 text-[10px] font-black flex items-center justify-center">{cartItems.length}</span>}
                    </Link>
                  </div>
                )}
              </div>

              {/* Fallback */}
              {fallback && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="mb-10 p-6 rounded-2xl bg-rose-950/30 border border-rose-900/50 backdrop-blur-lg">
                  <div className="flex items-start gap-4">
                    <AlertCircle className="w-8 h-8 text-rose-500 flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-xl font-bold text-rose-200 mb-2">Budget Matrix Incompatible</h3>
                      <p className="text-rose-100/70 mb-4">{fallback.message}</p>
                      <h4 className="font-mono text-xs uppercase tracking-widest text-rose-300 mb-3">Suggested Alternatives:</h4>
                      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {fallback.suggestedRentals.map((r, i) => (
                          <li key={i} className="bg-black/40 px-4 py-2 rounded-lg border border-rose-900/30 text-rose-100 flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-500" />{r}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Product Grid */}
              {/* Product Grid by Categories */}
              <div className="space-y-12">
                {['Top Search Products', 'Most Reliable', 'Best Seller', 'Most Discounted', 'Most Monthly Sales', 'Value for Money', 'Trending', 'Curated Node'].map((category) => {
                  const categoryItems = items.filter(it => it.tags?.includes(category) || (category === 'Curated Node' && (!it.tags || it.tags.length === 0)));
                  const completedItems = categoryItems.filter(it => it.status === 'complete');
                  const pendingCount = categoryItems.filter(it => it.status !== 'complete').length;

                  if (completedItems.length === 0 && pendingCount === 0) return null;

                  return (
                    <div key={category} className="space-y-6">
                      <div className="flex items-center gap-4">
                        <h3 className="text-xl font-bold tracking-tight text-white uppercase font-mono">{category}</h3>
                        <div className="h-px flex-grow bg-gradient-to-r from-slate-700 to-transparent" />
                      </div>

                      {pendingCount > 0 && (
                        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-4 p-5 bg-card/40 backdrop-blur-xl rounded-2xl border border-card-border shadow-xl overflow-hidden relative">
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/5 to-transparent animate-[shimmer_2s_infinite] -translate-x-[100%]" />
                          <div className="w-10 h-10 rounded-xl bg-slate-950 flex items-center justify-center border border-slate-800 shadow-inner">
                            <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
                          </div>
                          <div className="flex flex-col">
                            <span className="text-sm font-bold tracking-wide text-slate-200">Processing {pendingCount} topological {pendingCount === 1 ? 'node' : 'nodes'}</span>
                            <span className="text-xs font-mono text-slate-500">Extracting market data and computing reliability vectors...</span>
                          </div>
                        </motion.div>
                      )}

                      {completedItems.length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                          <AnimatePresence>
                            {completedItems.map((item, idx) => renderProductCard(item, idx))}
                          </AnimatePresence>
                        </div>
                      )}
                    </div>
                  );
                })}
                {/* Explore Further Trigger */}
                {items.length > 0 && !isProcessing && queryType === 'Indefinite Query' && (
                  <div className="mt-12 py-10 flex flex-col items-center gap-6 border-t border-white/5">
                    <div className="text-center space-y-2">
                      <h4 className="text-lg font-black text-white uppercase tracking-tighter">Reach the limit of current nodes?</h4>
                      <p className="text-xs text-slate-500 font-mono">My generative engine can extrapolate even deeper market dimensions.</p>
                    </div>
                    <button onClick={handleExploreFurther}
                      className="group relative px-10 py-4 rounded-2xl bg-white text-black font-black text-xs uppercase tracking-[0.2em] shadow-[0_0_40px_rgba(255,255,255,0.2)] hover:shadow-[0_0_60px_rgba(255,255,255,0.4)] transition-all overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                      <span className="relative z-10 group-hover:text-white flex items-center gap-2">
                        <Zap className="w-4 h-4 fill-current" /> Explore Further
                      </span>
                    </button>
                  </div>
                )}


              </div>
            </div>
          </motion.div>
        )}
        <AnimatePresence>
          {tunerOpen && <WeightTuner onClose={() => setTunerOpen(false)} />}
        </AnimatePresence>
      </AnimatePresence>
    </div>
  );
}
