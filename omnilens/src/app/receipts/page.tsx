'use client';

import { motion } from 'framer-motion';
import { useStore } from '@/store/useStore';
import { Receipt, Calendar, ArrowLeft, Trash2 } from 'lucide-react';
import Link from 'next/link';

export default function ReceiptsPage() {
    const receipts = useStore((s) => s.receipts || []);

    return (
        <div className="min-h-screen px-10 pt-18 pb-12 xl:px-16 bg-transparent">
            {/* Page Header */}
            <div className="mb-12">
                <div className="flex items-center gap-4 mb-3">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
                        <Receipt className="w-6 h-6 text-white drop-shadow-md" />
                    </div>
                    <div>
                        <h1 className="text-4xl font-bold text-foreground tracking-tight mb-2">
                            Order Manifests
                        </h1>
                        <p className="text-indigo-600 dark:text-indigo-400 font-mono text-[10px] md:text-xs uppercase tracking-[0.4em] font-bold opacity-80">Transaction History</p>
                    </div>
                </div>
                <div className="flex items-center justify-between ml-[64px]">
                    <p className="text-slate-500 dark:text-slate-400 font-mono text-sm tracking-wide flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                        {receipts.length} {receipts.length === 1 ? 'Digital Manifest' : 'Digital Manifests'} Compiled
                    </p>
                    {receipts.length > 0 && (
                        <button onClick={useStore.getState().clearReceipts} className="group flex items-center gap-3 text-[10px] font-mono font-bold text-text-muted hover:text-rose-600 dark:hover:text-rose-400 transition-all border border-card-border hover:border-rose-300 dark:hover:border-rose-900/50 px-6 py-3 rounded-2xl bg-card backdrop-blur-xl shadow-sm hover:shadow-lg">
                            <Trash2 className="w-4 h-4 group-hover:rotate-12 transition-transform" />
                            CLEAR MATRIX
                        </button>
                    )}
                </div>
            </div>

            {receipts.length === 0 ? (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center justify-center py-32 text-center">
                    <div className="w-20 h-20 rounded-3xl bg-card border border-card-border flex items-center justify-center mb-6 shadow-sm">
                        <Receipt className="w-10 h-10 text-text-muted opacity-50" />
                    </div>
                    <h2 className="text-xl font-bold text-slate-400 dark:text-slate-500 mb-2">No manifests found</h2>
                    <p className="text-slate-500 dark:text-slate-600 text-sm font-mono mb-8">
                        Complete a checkout sequence in your cart to generate a digital receipt.
                    </p>
                    <Link href="/" className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-colors font-bold tracking-widest uppercase text-xs font-mono shadow-lg">
                        <ArrowLeft className="w-4 h-4" /> Return to Core
                    </Link>
                </motion.div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {receipts.map((receipt, i) => (
                        <motion.div key={receipt.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
                            className="bg-card border border-card-border shadow-sm dark:shadow-none rounded-[2rem] p-6 flex flex-col hover:bg-white/50 dark:hover:bg-white/[0.04] transition-colors relative overflow-hidden group">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 dark:bg-indigo-500/10 rounded-full blur-3xl group-hover:bg-indigo-500/10 dark:group-hover:bg-indigo-500/20 transition-all pointer-events-none" />
                            <div className="flex justify-between items-start mb-6 border-b border-slate-100 dark:border-white/5 pb-4">
                                <div>
                                    <h3 className="text-sm font-mono font-bold text-indigo-500 dark:text-indigo-400 mb-1">{receipt.id}</h3>
                                    <div className="flex items-center gap-2 text-xs text-slate-400 dark:text-slate-500 font-mono">
                                        <Calendar className="w-3.5 h-3.5" />
                                        {new Date(receipt.date).toLocaleDateString()}
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <span className="text-lg font-black text-foreground">₹{receipt.total.toLocaleString()}</span>
                                    <button onClick={(e) => { e.preventDefault(); useStore.getState().removeReceipt(receipt.id); }} className="w-8 h-8 rounded-lg flex items-center justify-center bg-background/50 hover:bg-rose-500 text-text-muted hover:text-white transition-all transform hover:scale-105 shadow-sm border border-card-border"><Trash2 className="w-3.5 h-3.5" /></button>
                                </div>
                            </div>
                            <div className="space-y-3 flex-grow max-h-[150px] overflow-y-auto scrollbar-thin dark:scrollbar-thumb-slate-800 pr-2">
                                {receipt.items.map((item, idx) => (
                                    <div key={idx} className="flex justify-between items-start gap-4 text-sm">
                                        <span className="text-slate-600 dark:text-slate-300 line-clamp-2 leading-snug">{item.name}</span>
                                        <span className="text-slate-500 font-mono shrink-0">₹{item.price.toLocaleString()}</span>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
}
