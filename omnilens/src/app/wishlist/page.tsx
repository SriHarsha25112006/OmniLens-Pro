'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '@/store/useStore';
import { getApiUrl } from '@/lib/config';
import { Heart, ShoppingCart, ShoppingBag, Trash2 } from 'lucide-react';

export default function WishlistPage() {
    const wishlist = useStore((s) => s.wishlist);
    const toggleWishlist = useStore((s) => s.toggleWishlist);
    const clearWishlist = useStore((s) => s.clearWishlist);
    const addToCart = useStore((s) => s.addToCart);
    const cartItems = useStore((s) => s.cartItems);

    const [showClearConfirm, setShowClearConfirm] = useState(false);
    const [wishlistSuggestions, setWishlistSuggestions] = useState<any[]>([]);
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
    const prevWishlistLength = useRef(0);

    useEffect(() => {
        if (wishlist.length > 0 && wishlist.length !== prevWishlistLength.current) {
            prevWishlistLength.current = wishlist.length;
            setIsLoadingSuggestions(true);
            fetch(`${getApiUrl()}/api/wishlist_suggestions`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ wishlist: wishlist.filter(w => w.status === 'complete') })
            })
            .then(res => res.json())
            .then(data => { if (data.items) setWishlistSuggestions(data.items); })
            .catch(console.error)
            .finally(() => setIsLoadingSuggestions(false));
        } else if (wishlist.length === 0) {
            setWishlistSuggestions([]);
            prevWishlistLength.current = 0;
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [wishlist.length]);

    // wishlist now contains full ProductItem objects — no need to cross-reference items/allItems
    const wishlisted = wishlist.filter((item) => item.status === 'complete');

    return (
        <div className="min-h-screen px-10 pt-18 pb-12 xl:px-16">
            {/* Clear Confirm Modal */}
            <AnimatePresence>
                {showClearConfirm && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm px-4">
                        <motion.div initial={{ scale: 0.95, opacity: 0, y: 10 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0, y: 10 }}
                            className="bg-card border border-card-border rounded-3xl p-8 max-w-sm w-full shadow-2xl flex flex-col items-center text-center">
                            <div className="w-16 h-16 rounded-2xl bg-rose-500/20 text-rose-400 flex items-center justify-center mb-6">
                                <Trash2 className="w-8 h-8" />
                            </div>
                            <h2 className="text-xl font-bold text-white mb-2">Clear entire wishlist?</h2>
                            <p className="text-slate-400 text-sm mb-8">This action cannot be undone and will remove all saved items.</p>
                            <div className="flex gap-3 w-full">
                                <button onClick={() => setShowClearConfirm(false)} className="flex-1 px-4 py-3 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors font-bold text-sm">Cancel</button>
                                <button onClick={() => { clearWishlist(); setShowClearConfirm(false); }} className="flex-1 px-4 py-3 rounded-xl bg-rose-600 hover:bg-rose-500 text-white transition-colors font-bold text-sm shadow-[0_0_15px_rgba(225,29,72,0.4)]">Clear All</button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Page Header */}
            <div className="mb-12">
                <div className="flex items-center gap-4 mb-3">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center shadow-lg">
                        <Heart className="w-6 h-6 text-white fill-current drop-shadow-md" />
                    </div>
                    <div>
                        <h1 className="text-4xl font-bold text-foreground tracking-tight mb-2">
                            Wishlist
                        </h1>
                        <p className="text-rose-600 dark:text-rose-400 font-mono text-[10px] md:text-xs uppercase tracking-[0.4em] font-bold opacity-80">Curated Selection</p>
                    </div>
                </div>
                <div className="flex items-center justify-between ml-[64px]">
                    <p className="text-slate-500 dark:text-slate-400 font-mono text-sm tracking-wide flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                        {wishlisted.length} {wishlisted.length === 1 ? 'item' : 'items'} saved
                    </p>
                    {wishlisted.length > 0 && (
                        <button
                            onClick={() => setShowClearConfirm(true)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-600 hover:text-rose-400 hover:border-rose-900/50 transition-colors text-xs font-mono"
                        >
                            <Trash2 className="w-3.5 h-3.5" /> Clear All
                        </button>
                    )}
                </div>
            </div>

            {wishlisted.length === 0 ? (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-center justify-center py-32 text-center"
                >
                    <div className="w-20 h-20 rounded-3xl bg-card border border-card-border flex items-center justify-center mb-6">
                        <Heart className="w-10 h-10 text-text-muted opacity-50" />
                    </div>
                    <h2 className="text-xl font-bold text-slate-500 mb-2">Your wishlist is empty</h2>
                    <p className="text-slate-600 text-sm font-mono">
                        Click the ❤️ on any product card to save it here.
                    </p>
                </motion.div>
            ) : (
                <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-6 space-y-6">
                    <AnimatePresence>
                        {wishlisted.map((item, idx) => (
                            <motion.div
                                key={item.id}
                                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                transition={{ delay: idx * 0.05 }}
                                className="break-inside-avoid group relative"
                            >
                                <div className="absolute -inset-0.5 bg-gradient-to-br from-rose-500/20 to-pink-500/0 rounded-2xl blur opacity-0 group-hover:opacity-100 transition duration-500" />
                                <div className="relative bg-card backdrop-blur-xl border border-card-border rounded-2xl overflow-hidden shadow-2xl hover:-translate-y-1 transition-all duration-300 hover:border-rose-500/30">

                                    {/* Image */}
                                    {item.image && (
                                        <div className="relative h-44 w-full bg-white overflow-hidden p-3">
                                            <div className="absolute inset-0 bg-gradient-to-t from-slate-900/60 to-transparent z-10" />
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img
                                                src={item.image}
                                                alt={item.name}
                                                className="w-full h-full object-contain mix-blend-multiply"
                                            />
                                            {/* Platform badge */}
                                            <div className="absolute bottom-2 right-2 z-20">
                                                <span
                                                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full border backdrop-blur-md ${item.platform?.includes('Amazon')
                                                        ? 'bg-orange-500/20 border-orange-500/40 text-orange-300'
                                                        : 'bg-blue-500/20 border-blue-500/40 text-blue-300'
                                                        }`}
                                                >
                                                    {item.platform}
                                                </span>
                                            </div>
                                        </div>
                                    )}

                                    {/* Content */}
                                    <div className="p-4">
                                        <h3 className="font-semibold text-foreground mb-1 leading-tight group-hover:text-rose-300 transition-colors">
                                            {item.name}
                                        </h3>
                                        {item.score !== undefined && (
                                            <p className="text-[11px] text-slate-500 font-mono mb-3">
                                                ML Score: {item.score.toFixed(1)}/100 | AI Recommended
                                            </p>
                                        )}
                                        <div className="flex items-center justify-between mb-4">
                                            <span className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                                                ₹{(item.finalPrice || 0).toLocaleString()}
                                            </span>
                                        </div>

                                        {/* Actions */}
                                        <div className="flex gap-2">
                                            {item.external_link && (
                                                <a
                                                    href={item.external_link}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="flex-1 flex items-center justify-center gap-1.5 bg-white text-black px-3 py-2 rounded-lg font-bold text-xs hover:bg-slate-100 transition-all shadow-sm"
                                                >
                                                    <ShoppingBag className="w-3.5 h-3.5" /> BUY
                                                </a>
                                            )}
                                            <button
                                                onClick={() => addToCart(item)}
                                                className={`flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg font-bold text-xs border transition-all ${cartItems.find((c) => c.id === item.id)
                                                    ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                                                    : 'border-slate-700 text-slate-400 hover:border-emerald-500/50 hover:text-emerald-300'
                                                    }`}
                                            >
                                                <ShoppingCart className="w-3.5 h-3.5" />
                                                {cartItems.find((c) => c.id === item.id) ? 'Added' : 'Cart'}
                                            </button>
                                            <button
                                                onClick={() => toggleWishlist(item)}
                                                title="Remove from wishlist"
                                                className="flex items-center justify-center w-9 rounded-lg border border-rose-900/50 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 transition-colors"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            )}

            {/* Wishlist Suggestions */}
            {wishlisted.length > 0 && (wishlistSuggestions.length > 0 || isLoadingSuggestions) && (
                <div className="mt-16 pt-12 border-t border-slate-800">
                    <div className="flex items-center gap-4 mb-8">
                        <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-amber-500 to-rose-500 flex items-center justify-center shadow-[0_0_20px_rgba(245,158,11,0.3)]">
                            <span className="text-lg">✨</span>
                        </div>
                        <h2 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-amber-200 to-rose-400 tracking-tight">
                            Suggested Upgrades
                        </h2>
                        {isLoadingSuggestions && (
                            <div className="ml-4 flex gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                                <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse delay-75" />
                                <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse delay-150" />
                            </div>
                        )}
                    </div>
                    
                    <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-6 space-y-6">
                        <AnimatePresence>
                            {wishlistSuggestions.map((item, idx) => (
                                <motion.div
                                    key={item.id}
                                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.9 }}
                                    transition={{ delay: idx * 0.05 }}
                                    className="break-inside-avoid group relative"
                                >
                                    <div className="absolute -inset-0.5 bg-gradient-to-br from-amber-500/20 to-rose-500/0 rounded-2xl blur opacity-0 group-hover:opacity-100 transition duration-500" />
                                    <div className="relative bg-card/40 backdrop-blur-xl border border-card-border rounded-2xl overflow-hidden shadow-2xl hover:-translate-y-1 transition-all duration-300 hover:border-amber-500/30">
                                        
                                        {item.suggested_for && (
                                            <div className="absolute top-2 left-2 z-30">
                                                <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-amber-500/20 border-amber-500/40 text-amber-300 backdrop-blur-md">
                                                    For: {item.suggested_for.length > 20 ? item.suggested_for.substring(0,20)+'...' : item.suggested_for}
                                                </span>
                                            </div>
                                        )}

                                        {/* Image */}
                                        {item.image && (
                                            <div className="relative h-44 w-full bg-white overflow-hidden p-3 pt-8">
                                                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/60 to-transparent z-10" />
                                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                                <img
                                                    src={item.image}
                                                    alt={item.name}
                                                    className="w-full h-full object-contain mix-blend-multiply transition-transform duration-500 group-hover:scale-105"
                                                />
                                                <div className="absolute bottom-2 right-2 z-20">
                                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border backdrop-blur-md ${item.platform?.includes('Amazon') ? 'bg-orange-500/20 border-orange-500/40 text-orange-300' : 'bg-blue-500/20 border-blue-500/40 text-blue-300'}`}>
                                                        {item.platform}
                                                    </span>
                                                </div>
                                            </div>
                                        )}

                                        {/* Content */}
                                        <div className="p-4 pt-4 bg-card relative z-20 border-t border-card-border">
                                            <h3 className="font-semibold text-foreground mb-1 leading-tight group-hover:text-amber-500 transition-colors line-clamp-2">
                                                {item.name}
                                            </h3>
                                            {item.score !== undefined && (
                                                <p className="text-[11px] text-slate-500 font-mono mb-3">
                                                    Upgrade Score: {item.score.toFixed(1)}/100
                                                </p>
                                            )}
                                            
                                            <div className="flex items-center justify-between mb-4 mt-auto">
                                                <span className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 to-cyan-500">
                                                    ₹{(item.finalPrice || 0).toLocaleString()}
                                                </span>
                                            </div>

                                            {/* Actions */}
                                            <div className="flex gap-2">
                                                {item.external_link && (
                                                    <a href={item.external_link} target="_blank" rel="noopener noreferrer"
                                                        className="flex-1 flex items-center justify-center gap-1.5 bg-white text-black px-3 py-2 rounded-lg font-bold text-xs hover:bg-slate-100 transition-all shadow-sm">
                                                        <ShoppingBag className="w-3.5 h-3.5" /> BUY
                                                    </a>
                                                )}
                                                <button onClick={() => addToCart(item)}
                                                    className={`flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg font-bold text-xs border transition-all ${cartItems.find((c) => c.id === item.id) ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300' : 'dark:border-slate-700 dark:text-slate-400 border-slate-300 text-slate-600 hover:border-emerald-500/50 hover:text-emerald-500'}`}>
                                                    <ShoppingCart className="w-3.5 h-3.5" />
                                                    {cartItems.find((c) => c.id === item.id) ? 'Added' : 'Cart'}
                                                </button>
                                                <button onClick={() => toggleWishlist(item)} title="Add to Wishlist"
                                                    className="flex items-center justify-center w-9 rounded-lg border dark:border-slate-700 dark:bg-slate-800 border-slate-300 bg-slate-100 text-rose-500 hover:bg-rose-500 hover:text-white hover:border-rose-500 transition-colors">
                                                    <Heart className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                </div>
            )}
        </div>
    );
}
