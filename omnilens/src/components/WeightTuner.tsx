'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Settings2, SlidersHorizontal, Info } from 'lucide-react';

interface WeightTunerProps {
    onClose: () => void;
}

export default function WeightTuner({ onClose }: WeightTunerProps) {
    // Note: In a real app, these would be bound to the store and trigger a re-render/re-score
    const [weights, setWeights] = React.useState({
        quality: 40,
        price: 30,
        volume: 20,
        brand: 10
    });

    return (
        <motion.div
            initial={{ opacity: 0, x: 20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.95 }}
            className="fixed top-24 right-6 z-50 w-72 glass-panel rounded-3xl p-6 shadow-2xl shadow-purple-500/10 border border-white/10"
        >
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <Settings2 className="w-5 h-5 text-purple-400" />
                    <h3 className="font-bold text-base tracking-tight text-white">Scorer Matrix</h3>
                </div>
                <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
                    <SlidersHorizontal className="w-4 h-4" />
                </button>
            </div>

            <div className="space-y-6">
                {Object.entries(weights).map(([key, value]) => (
                    <div key={key} className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-[10px] uppercase font-mono tracking-widest text-slate-400">{key} Bias</span>
                            <span className="text-xs font-bold text-purple-400">{value}%</span>
                        </div>
                        <div className="relative h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${value}%` }}
                                className="absolute inset-y-0 left-0 bg-gradient-to-r from-purple-500 to-cyan-400"
                            />
                            <input
                                type="range"
                                min="0"
                                max="100"
                                value={value}
                                onChange={(e) => setWeights({ ...weights, [key]: parseInt(e.target.value) })}
                                className="absolute inset-0 opacity-0 cursor-pointer"
                            />
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-8 pt-4 border-t border-white/5 flex items-start gap-2">
                <Info className="w-3.5 h-3.5 text-slate-500 mt-0.5" />
                <p className="text-[10px] text-slate-500 leading-relaxed italic">
                    Adjusting these vectors will re-calculate product reliability in real-time.
                </p>
            </div>

            <button
                onClick={onClose}
                className="w-full mt-6 py-2.5 rounded-xl bg-purple-600/20 border border-purple-500/30 text-purple-300 text-xs font-bold hover:bg-purple-600/40 transition-all"
            >
                Deploy Weights
            </button>
        </motion.div>
    );
}
