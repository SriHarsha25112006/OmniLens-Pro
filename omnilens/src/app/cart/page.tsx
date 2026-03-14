'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '@/store/useStore';
import { getApiUrl } from '@/lib/config';
import { ShoppingCart, Trash2, Receipt, ShoppingBag, ArrowLeft } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function CartPage() {
    const cartItems = useStore((s) => s.cartItems);
    const removeFromCart = useStore((s) => s.removeFromCart);
    const clearCart = useStore((s) => s.clearCart);
    const addReceipt = useStore((s) => s.addReceipt);
    const setItems = useStore((s) => s.setItems);
    const setAllItems = useStore((s) => s.setAllItems);
    const setStatusMessage = useStore((s) => s.setStatusMessage);
    const clearChat = useStore((s) => s.clearChat);

    const [confirmed, setConfirmed] = useState(false);
    const [countdown, setCountdown] = useState(3);
    const [showClearConfirm, setShowClearConfirm] = useState(false);
    const router = useRouter();

    const cartTotal = cartItems.reduce((acc, i) => acc + (i.finalPrice || 0), 0);

    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (confirmed && countdown > 0) {
            timer = setTimeout(() => setCountdown(countdown - 1), 1000);
        } else if (confirmed && countdown === 0) {
            (async () => {
                try {
                    await fetch(`${getApiUrl()}/api/clear_session`, { method: 'POST' });
                } catch (e) {
                    console.error("Failed to clear backend session:", e);
                }
            })();

            clearCart();
            clearChat();
            setItems([]);
            setAllItems([]);
            setStatusMessage('Awaiting prompt sequence...');
            router.push('/');
        }
        return () => clearTimeout(timer);
    }, [confirmed, countdown, clearCart, clearChat, setItems, setAllItems, setStatusMessage, router]);

    const handleConfirm = () => {
        if (cartItems.length === 0) return;
        const newReceipt = { id: `RCPT-${Math.floor(Math.random() * 100000).toString().padStart(5, '0')}`, date: new Date().toISOString(), total: cartTotal, items: cartItems.map(i => ({ name: i.name, price: i.finalPrice || 0 })) };
        addReceipt(newReceipt);
        setConfirmed(true);
    };

    return (
        <div className="min-h-screen px-10 pt-18 pb-12 xl:px-16 relative overflow-hidden bg-transparent">
            {/* Ambient Background */}
            <div className="absolute inset-0 pointer-events-none">
               <motion.div animate={{ scale: [1, 1.1, 1], opacity: [0.1, 0.2, 0.1] }} transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
                 className="absolute top-0 right-0 w-[500px] h-[500px] rounded-full bg-[radial-gradient(circle,rgba(52,211,153,0.15)_0%,rgba(6,182,212,0.05)_50%,transparent_70%)] blur-[80px]" />
               <motion.div animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.15, 0.1] }} transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 2 }}
                 className="absolute bottom-0 left-0 w-[600px] h-[600px] rounded-full bg-[radial-gradient(circle,rgba(168,85,247,0.1)_0%,rgba(236,72,153,0.05)_50%,transparent_70%)] blur-[100px]" />
            </div>

            <div className="relative z-10 w-full">
                {/* Clear Confirm Modal */}
                <AnimatePresence>
                    {showClearConfirm && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md px-4">
                            <motion.div initial={{ scale: 0.95, opacity: 0, y: 10 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0, y: 10 }}
                                className="bg-card border border-card-border rounded-[2rem] p-8 max-w-sm w-full shadow-2xl flex flex-col items-center text-center">
                                <div className="w-16 h-16 rounded-2xl bg-rose-100 dark:bg-rose-500/20 text-rose-500 flex items-center justify-center mb-6 shadow-sm">
                                    <Trash2 className="w-8 h-8" />
                                </div>
                                <h2 className="text-xl font-bold text-slate-800 dark:text-white mb-2">Clear entire cart?</h2>
                                <p className="text-slate-500 dark:text-slate-400 text-sm mb-8">This action cannot be undone and will remove all items from your cart matrix.</p>
                                <div className="flex gap-3 w-full">
                                    <button onClick={() => setShowClearConfirm(false)} className="flex-1 px-4 py-3 rounded-xl border border-card-border text-text-muted hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors font-bold text-sm">Cancel</button>
                                    <button onClick={() => { clearCart(); setShowClearConfirm(false); }} className="flex-1 px-4 py-3 rounded-xl bg-gradient-to-r from-rose-500 to-rose-600 hover:from-rose-600 hover:to-rose-700 text-white transition-colors font-bold text-sm shadow-lg">Clear All</button>
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Thank You Overlay */}
                <AnimatePresence>
                    {confirmed && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background before:absolute before:inset-0 before:bg-[radial-gradient(ellipse_at_center,rgba(52,211,153,0.15)__0%,transparent_100%)]">
                            <motion.div initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                                className="w-24 h-24 rounded-[2rem] bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-[0_0_60px_rgba(52,211,153,0.6)] mb-8 relative">
                                <div className="absolute inset-0 bg-white/20 rounded-[2rem] blur-md animate-pulse" />
                                <Receipt className="w-12 h-12 text-white relative z-10 drop-shadow-md" />
                            </motion.div>
                            <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
                                className="text-5xl md:text-7xl font-black text-foreground mb-4 text-center drop-shadow-lg tracking-tight">
                                Trace Confirmed
                            </motion.h1>
                            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
                                className="text-emerald-600 dark:text-emerald-400 font-mono tracking-widest text-sm md:text-base mb-10 uppercase font-bold text-center">
                                Initializing Order Sequence
                            </motion.p>

                            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
                                className="bg-card border border-card-border rounded-[2rem] p-6 md:p-8 max-w-md w-full mb-10 backdrop-blur-2xl relative overflow-hidden shadow-2xl">
                                <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-emerald-400 to-cyan-400" />
                                <div className="flex justify-between items-end mb-6 border-b border-slate-200 dark:border-slate-800 pb-5 text-slate-800 dark:text-white">
                                    <h3 className="text-xl font-bold">Digital Receipt</h3>
                                    <p className="font-mono text-xs">Sum: <span className="text-emerald-600 dark:text-emerald-400 font-bold text-sm md:text-base">₹{cartTotal.toLocaleString()}</span></p>
                                </div>
                                <div className="space-y-4 max-h-[180px] overflow-y-auto scrollbar-thin dark:scrollbar-thumb-slate-800 pr-2">
                                    {cartItems.map((item, i) => (
                                        <div key={i} className="flex justify-between items-center text-sm md:text-base">
                                            <span className="text-slate-600 dark:text-slate-300 truncate max-w-[70%] font-medium">{item.name}</span>
                                            <span className="text-slate-500 font-mono">₹{(item.finalPrice || 0).toLocaleString()}</span>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>

                            <div className="flex flex-col items-center">
                                <motion.div key={countdown} initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1.5, opacity: 1 }} transition={{ duration: 0.5 }}
                                    className="text-5xl font-black text-emerald-500 dark:text-emerald-400 mb-3 drop-shadow-md">
                                    {countdown > 0 ? countdown : '🚀'}
                                </motion.div>
                                <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
                                    className="text-slate-500 dark:text-slate-600 text-[10px] md:text-xs font-mono tracking-widest uppercase font-bold">
                                    Returning to Nexus...
                                </motion.p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Header */}
                <div className="mb-12">
                    <div className="flex items-center gap-4 mb-3">
                        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg group-hover:shadow-emerald-500/20 transition-all">
                            <ShoppingCart className="w-6 h-6 text-white drop-shadow-md" />
                        </div>
                        <div>
                            <h1 className="text-4xl font-bold text-foreground tracking-tight mb-2">My Cart</h1>
                            <p className="text-emerald-600 dark:text-emerald-400 font-mono text-[10px] md:text-xs uppercase tracking-[0.4em] font-bold opacity-80">Operational Nodes</p>
                        </div>
                    </div>
                    
                    <div className="flex items-center justify-between ml-[64px]">
                        <p className="text-slate-500 dark:text-slate-400 font-mono text-sm tracking-wide font-medium flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                            {cartItems.length} {cartItems.length === 1 ? 'Integrated Unit' : 'Integrated Units'} · <span className="text-emerald-600 dark:text-emerald-400 font-bold">₹{cartTotal.toLocaleString()}</span> Transmission Value
                        </p>
                        
                        {cartItems.length > 0 && (
                            <button onClick={() => setShowClearConfirm(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-600 hover:text-rose-400 hover:border-rose-900/50 transition-colors text-xs font-mono uppercase tracking-wider bg-transparent">
                                <Trash2 className="w-3.5 h-3.5" /> Clear Matrix
                            </button>
                        )}
                    </div>
                </div>

                {cartItems.length === 0 ? (
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center justify-center py-32 text-center">
                    <div className="w-20 h-20 rounded-3xl bg-card border border-card-border flex items-center justify-center mb-6 shadow-sm">
                            <ShoppingCart className="w-10 h-10 text-slate-400 dark:text-slate-700" />
                        </div>
                        <h2 className="text-xl font-bold text-slate-400 dark:text-slate-500 mb-2">Your cart is empty</h2>
                        <p className="text-slate-500 dark:text-slate-600 text-sm font-mono mb-8">
                            Engage the <ShoppingCart className="w-3.5 h-3.5 inline mx-1 flex-shrink-0 text-emerald-500" /> on any product card to integrate it.
                        </p>
                        <button onClick={() => router.push('/')} className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-colors font-bold tracking-widest uppercase text-xs font-mono shadow-lg">
                            <ArrowLeft className="w-4 h-4" /> Return to Core
                        </button>
                    </motion.div>
                ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 md:gap-10">
                        <div className="lg:col-span-8 space-y-4">
                            <AnimatePresence>
                                {cartItems.map((item, i) => (
                                    <motion.div key={item.id} initial={{ opacity: 0, x: -20, scale: 0.98 }} animate={{ opacity: 1, x: 0, scale: 1 }} exit={{ opacity: 0, x: 20, scale: 0.95 }} transition={{ delay: i * 0.05, type: 'spring', stiffness: 200, damping: 20 }}
                                        className="flex flex-col sm:flex-row items-start sm:items-center gap-4 sm:gap-6 p-5 sm:p-6 bg-card backdrop-blur-xl border border-card-border rounded-[2rem] hover:border-emerald-500/30 dark:hover:border-emerald-500/50 transition-all group shadow-sm hover:shadow-xl">
                                        <div className="flex items-center gap-4 w-full sm:w-auto">
                                           <span className="text-slate-300 dark:text-slate-600 font-mono text-xs font-bold w-4 shrink-0">0{i + 1}</span>
                                           {item.image && (
                                              <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-white flex-shrink-0 overflow-hidden p-1 sm:p-2 border border-slate-100 dark:border-white/5 shadow-inner">
                                                  <img src={item.image} alt={item.name} className="w-full h-full object-contain mix-blend-multiply" />
                                              </div>
                                           )}
                                        </div>
                                        <div className="flex-grow min-w-0 w-full">
                                            <p className="font-bold text-base sm:text-lg text-slate-800 dark:text-slate-200 truncate group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors mb-1">{item.name}</p>
                                            <div className="flex items-center gap-3">
                                               <span className="px-2 py-1 bg-slate-100 dark:bg-white/5 rounded text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase tracking-widest font-bold">{item.platform}</span>
                                               {item.score !== undefined && (
                                                   <span className="text-[10px] text-emerald-600 dark:text-emerald-500 font-mono font-bold">NODE RATING: {item.score.toFixed(1)}/100</span>
                                               )}
                                            </div>
                                        </div>
                                        <div className="flex items-center justify-between w-full sm:w-auto gap-6 sm:gap-4 shrink-0 mt-4 sm:mt-0 pt-4 sm:pt-0 border-t sm:border-0 border-slate-200 dark:border-white/5">
                                            <span className="text-xl sm:text-2xl font-black text-emerald-600 dark:text-emerald-400 drop-shadow-sm">₹{(item.finalPrice || 0).toLocaleString()}</span>
                                            <div className="flex gap-2 items-center">
                                                {item.external_link && item.external_link !== '#' && (
                                                    <a href={item.external_link} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1.5 px-4 sm:px-5 py-2.5 rounded-xl bg-orange-500/10 dark:bg-orange-500/20 text-orange-600 dark:text-orange-400 border border-orange-500/30 hover:bg-orange-500 hover:text-white transition-all font-black text-xs tracking-widest uppercase shadow-sm">BUY NOW</a>
                                                )}
                                                <button onClick={() => removeFromCart(item.id)} className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-100 dark:bg-white/5 text-slate-400 hover:text-white hover:bg-rose-500 transition-all"><Trash2 className="w-4 h-4" /></button>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>

                        <div className="lg:col-span-4">
                            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                                className="sticky top-24 bg-card backdrop-blur-3xl border border-card-border rounded-[2.5rem] overflow-hidden shadow-2xl">
                                <div className="px-8 py-6 border-b border-slate-200 dark:border-white/5 bg-slate-50 dark:bg-black/40 flex items-center gap-3">
                                    <Receipt className="w-6 h-6 text-emerald-500" />
                                    <span className="text-lg font-black text-slate-800 dark:text-white tracking-tight">Transmission Summary</span>
                                </div>
                                <div className="p-8 space-y-4">
                                    {cartItems.map((item) => (
                                        <div key={item.id} className="flex justify-between items-center text-sm group">
                                            <span className="text-slate-600 dark:text-slate-400 font-medium truncate max-w-[65%] group-hover:text-slate-800 dark:group-hover:text-slate-200 transition-colors">{item.name}</span>
                                            <span className="text-slate-700 dark:text-slate-300 font-mono font-bold">₹{(item.finalPrice || 0).toLocaleString()}</span>
                                        </div>
                                    ))}
                                    <div className="h-px bg-slate-200 dark:bg-white/10 my-6 relative flex items-center justify-center">
                                       <div className="absolute px-2 bg-white dark:bg-slate-900 text-[10px] text-slate-400 font-mono uppercase tracking-widest">+ TAXES DEFERRED</div>
                                    </div>
                                    <div className="flex justify-between items-end mb-2">
                                        <span className="text-slate-500 dark:text-slate-400 font-mono text-xs uppercase tracking-widest font-bold">Total Metric</span>
                                        <span className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-cyan-500 dark:from-emerald-400 dark:to-cyan-400 drop-shadow-sm">₹{cartTotal.toLocaleString()}</span>
                                    </div>
                                </div>
                                <div className="px-8 pb-8 space-y-4">
                                    <button onClick={handleConfirm} className="w-full py-4 rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-white font-black tracking-[0.2em] uppercase text-sm transition-all shadow-lg hover:-translate-y-1">Confirm Protocol</button>
                                    <button onClick={() => router.push('/')} className="w-full py-3.5 rounded-2xl border-2 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 transition-all font-mono text-xs font-bold tracking-widest flex items-center justify-center gap-2 hover:bg-slate-50 dark:hover:bg-transparent shadow-sm"><ArrowLeft className="w-4 h-4" /> RE-ENTER MATRIX</button>
                                </div>
                            </motion.div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
