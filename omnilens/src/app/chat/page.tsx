'use client';

import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '@/store/useStore';
import { getApiUrl } from '@/lib/config';
import { Bot, Send, Trash2, Terminal, ChevronLeft } from 'lucide-react';
import Link from 'next/link';

export default function ChatPage() {
    const chatMessages = useStore((s) => s.chatMessages);
    const addChatMessage = useStore((s) => s.addChatMessage);
    const clearChat = useStore((s) => s.clearChat);
    const items = useStore((s) => s.items);
    const popHistory = useStore((s) => s.popHistory);
    const resetItems = useStore((s) => s.resetItems);
    const pushHistory = useStore((s) => s.pushHistory);
    const setItems = useStore((s) => s.setItems);

    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => { 
        endRef.current?.scrollIntoView({ behavior: 'smooth' }); 
    }, [chatMessages]);

    const send = async () => {
        const msg = input.trim();
        if (!msg || loading) return;
        setInput('');
        setLoading(true);
        addChatMessage({ id: Date.now().toString(), role: 'user', content: msg });
        const ml = msg.toLowerCase();

        // Rollback/Reset commands
        if (['undo', 'rollback', 'revert'].some(t => ml.includes(t))) {
            popHistory();
            addChatMessage({ id: Date.now().toString(), role: 'ai', content: '🔄 *Temporal shift successful.* Data state reverted to previous stable node.' });
            setLoading(false); return;
        }
        if (['reset', 'clear session'].some(t => ml.includes(t))) {
            resetItems();
            addChatMessage({ id: Date.now().toString(), role: 'ai', content: '🌊 *System flush complete.* All workspace parameters have been normalized.' });
            setLoading(false); return;
        }

        try {
            const res = await fetch(`${getApiUrl()}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, items: items.map(i => ({ id: i.id, name: i.name, finalPrice: i.finalPrice })) }),
            });
            const data = await res.json();

            if (data.action === 'remove' && data.remove_id) {
                pushHistory();
                setItems(items.filter(i => i.id !== data.remove_id));
            } else if (data.action === 'replace') {
                pushHistory();
                const ni = { ...data.new_item, status: 'complete' as const, progress: 100, category: 'Components' };
                if (data.remove_id) setItems(items.map(i => i.id === data.remove_id ? ni : i));
                else setItems([...items, ni]);
            } else if (data.action === 'add' && data.new_item) {
                pushHistory();
                setItems([...items, { ...data.new_item, status: 'complete' as const, progress: 100, category: 'Components' }]);
            }

            addChatMessage({ id: Date.now().toString(), role: 'ai', content: data.message || '✅ *Instruction executed.* Market nodes updated.' });
        } catch {
            addChatMessage({ id: Date.now().toString(), role: 'ai', content: '❌ *Critical: Uplink failed.* Ensure ML Engine is active.' });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex-1 flex flex-col bg-transparent overflow-hidden relative">
            {/* Ambient Background */}
            <div className="absolute inset-0 pointer-events-none z-0">
               <motion.div animate={{ scale: [1, 1.05, 1], opacity: [0.15, 0.25, 0.15] }} transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
                 className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] sm:w-[800px] sm:h-[800px] rounded-full bg-[radial-gradient(circle,rgba(168,85,247,0.15)_0%,rgba(6,182,212,0.05)_50%,transparent_70%)] blur-[80px]" />
            </div>

            {/* Messages Area */}
            <main className="flex-1 overflow-y-auto relative z-10 scroll-smooth pt-18 pb-12">
                <div className="max-w-5xl mx-auto w-full px-4 md:px-8 pb-10 space-y-8">
                    
                    {/* Integrated Header - Guaranteed Visibility */}
                    <div className="mb-10 p-6 rounded-[2rem] bg-card border-2 border-card-border shadow-xl ml-4 mr-4 md:ml-0">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-cyan-400 p-[1px] shadow-lg">
                                <div className="w-full h-full bg-card rounded-[11px] flex items-center justify-center">
                                    <Bot className="w-6 h-6 text-purple-600 dark:text-cyan-400" />
                                </div>
                            </div>
                            <div>
                                <h1 className="text-4xl font-bold text-foreground tracking-tight leading-none mb-2">
                                    Assistant Neural Core
                                </h1>
                                <div className="flex items-center gap-2 mt-1">
                                    <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
                                        <span className="text-[10px] font-mono font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest">Neural Stream Active</span>
                                    </div>
                                </div>
                            </div>
                            <div className="ml-auto">
                                <button onClick={clearChat} title="Clear Session" 
                                    className="p-4 rounded-2xl border border-slate-200 dark:border-white/10 text-slate-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-500/10 transition-all">
                                    <Trash2 className="w-6 h-6" />
                                </button>
                            </div>
                        </div>
                    </div>

                    {chatMessages.length === 0 && (
                        <div className="h-[50vh] flex flex-col items-center justify-center opacity-40 select-none">
                            <Bot className="w-20 h-20 mb-6 text-purple-600 dark:text-slate-500 drop-shadow-md" />
                            <p className="text-sm font-mono uppercase tracking-[0.3em] font-bold text-slate-500 dark:text-slate-400">Initialize Sequence</p>
                        </div>
                    )}

                    <AnimatePresence initial={false}>
                        {chatMessages.map((msg) => (
                            <motion.div key={msg.id} initial={{ opacity: 0, y: 20, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                                className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                
                                {msg.role === 'ai' && (
                                   <div className="flex-shrink-0 mr-3 self-end mb-6">
                                      <div className="w-10 h-10 rounded-full bg-white border border-slate-200 dark:bg-slate-800 dark:border-white/10 flex items-center justify-center shadow-lg">
                                        <Bot className="w-5 h-5 text-purple-600 dark:text-cyan-400" />
                                      </div>
                                   </div>
                                )}
                                
                                <div className={`relative max-w-[90%] md:max-w-[75%]`}>
                                    <div className={`rounded-3xl px-6 py-5 text-sm sm:text-base leading-relaxed border transition-all ${
                                        msg.role === 'user'
                                            ? 'bg-gradient-to-br from-purple-600 to-cyan-600 text-white rounded-br-md border-transparent shadow-lg shadow-purple-500/10'
                                            : 'bg-card/90 backdrop-blur-xl border-card-border text-foreground rounded-bl-md shadow-sm'
                                        }`} style={{ whiteSpace: 'pre-line' }}>
                                        {msg.content}
                                    </div>
                                    <div className={`mt-2 text-[10px] font-mono tracking-widest text-slate-500 dark:text-slate-400 ${msg.role === 'user' ? 'text-right mr-2' : 'text-left ml-2'}`}>
                                        {msg.role === 'user' ? 'USER_INPUT' : 'SYSTEM_RESPONSE'}
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {loading && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start items-end mb-4">
                            <div className="w-10 h-10 rounded-full bg-white border border-slate-200 dark:bg-slate-800/80 dark:border-white/10 flex items-center justify-center mr-3 mb-6 shadow-lg">
                                <Bot className="w-5 h-5 text-purple-600 dark:text-cyan-400" />
                            </div>
                            <div className="bg-card/80 dark:bg-card/10 border border-card-border rounded-[1.5rem] rounded-bl-md px-6 py-5 flex items-center gap-2 shadow-sm h-[60px] mb-6">
                                {[0, 1, 2].map((i) => (
                                    <motion.div key={i} className="w-2 h-2 rounded-full bg-purple-500 dark:bg-cyan-400/80" 
                                      animate={{ opacity: [0.3, 1, 0.3], y: [0, -5, 0] }} transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }} />
                                ))}
                            </div>
                        </motion.div>
                    )}
                    <div ref={endRef} className="h-4" />
                </div>
            </main>

            {/* Input Console */}
            <footer className="relative z-20 border-t border-card-border bg-nav-bg backdrop-blur-xl">
                <div className="max-w-5xl mx-auto w-full px-4 md:px-8 py-6">
                    <div className="relative flex items-end gap-3 bg-card border border-card-border rounded-2xl p-2 sm:p-3 shadow-inner shadow-slate-200 dark:shadow-none focus-within:border-purple-500/50 dark:focus-within:border-cyan-500/50 transition-all duration-300">
                        <div className="flex-shrink-0 p-3 self-center hidden sm:block">
                           <Terminal className="w-6 h-6 text-slate-400 dark:text-slate-500" />
                        </div>
                        <textarea value={input} onChange={e => setInput(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                            placeholder="Message OmniLens Core..."
                            rows={1}
                            disabled={loading}
                            className="flex-1 bg-transparent py-3 md:py-4 px-3 sm:px-0 text-sm md:text-base text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 outline-none resize-none leading-relaxed max-h-40 min-h-[56px] scrollbar-thin disabled:opacity-50" />
                        <button onClick={send} disabled={loading || !input.trim()}
                            className="w-14 h-14 flex-shrink-0 rounded-xl bg-gradient-to-br from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 disabled:from-slate-300 disabled:to-slate-300 text-white flex items-center justify-center shadow-lg transition-all transform hover:scale-105 active:scale-95 disabled:hover:scale-100 disabled:shadow-none group">
                            <Send className="w-6 h-6 ml-1 group-hover:-translate-y-1 group-hover:translate-x-1 transition-transform" />
                        </button>
                    </div>
                    <div className="text-center mt-3 flex items-center justify-center gap-6 text-slate-500 text-[10px] font-mono uppercase tracking-[0.2em]">
                        <p>↵ SEND</p>
                        <p>⇧+↵ NEW LINE</p>
                    </div>
                </div>
            </footer>
        </div>
    );
}
