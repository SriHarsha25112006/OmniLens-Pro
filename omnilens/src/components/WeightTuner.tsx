'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings2, RotateCcw, Zap, Info, Loader2, CheckCircle2, Brain } from 'lucide-react';
import { getApiUrl } from '@/lib/config';

interface FeatureImportances {
  model_importances: Record<string, number>;
  rlhf_bias: Record<string, number>;
  effective: Record<string, number>;
}

interface WeightTunerProps {
  onClose: () => void;
}

// Human-readable labels for LTR feature names
const FEATURE_LABELS: Record<string, { label: string; icon: string; description: string }> = {
  semantic_sim:  { label: 'Relevance',      icon: '🎯', description: 'How well the product matches your query (BGE semantic similarity)' },
  rating:        { label: 'Star Rating',    icon: '⭐', description: 'Normalized product star rating' },
  review_count:  { label: 'Review Volume',  icon: '💬', description: 'Log-scaled number of reviews' },
  sentiment:     { label: 'Sentiment',      icon: '❤️', description: 'RoBERTa customer review sentiment score' },
  brand_trust:   { label: 'Brand Trust',    icon: '🏆', description: 'Whether a recognized brand is detected' },
  discount:      { label: 'Discount',       icon: '🏷️', description: 'Discount percentage value' },
  sales_volume:  { label: 'Sales Volume',   icon: '📈', description: 'Log-scaled monthly sales count' },
  reliability:   { label: 'Reliability',    icon: '🔒', description: 'Composite data-density score' },
};

// Slider keys the backend understands (maps to LTR feature names internally)
const SLIDER_FEATURES = ['semantic_sim', 'rating', 'sentiment', 'brand_trust', 'discount', 'sales_volume'];

export default function WeightTuner({ onClose }: WeightTunerProps) {
  const [importances, setImportances] = useState<FeatureImportances | null>(null);
  const [sliderValues, setSliderValues] = useState<Record<string, number>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Load LTR importances from backend on mount
  const loadImportances = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch(getApiUrl() + '/api/feature_importances');
      if (!res.ok) throw new Error('Failed to fetch');
      const data: FeatureImportances = await res.json();
      setImportances(data);
      // Initialize sliders from effective weights (model + any existing RLHF bias)
      const initSliders: Record<string, number> = {};
      SLIDER_FEATURES.forEach((f) => {
        initSliders[f] = Math.round((data.effective[f] ?? 0) * 100);
      });
      setSliderValues(initSliders);
    } catch (e) {
      setStatusMsg({ text: 'Could not connect to ML engine', type: 'error' });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { loadImportances(); }, [loadImportances]);

  const hasBias = importances
    ? Object.values(importances.rlhf_bias).some((v) => Math.abs(v) > 0.001)
    : false;

  // Deploy: send slider deltas as RLHF bias
  const handleDeploy = async () => {
    setIsSaving(true);
    setStatusMsg(null);
    try {
      // Convert slider % → [0,1] and build generic slider key map
      const sliderMap: Record<string, number> = {
        price:      (sliderValues['discount']    ?? 10) / 100,
        rating:     (sliderValues['rating']      ?? 20) / 100,
        sentiment:  (sliderValues['sentiment']   ?? 20) / 100,
        bestseller: (sliderValues['brand_trust'] ?? 10) / 100,
        sales:      (sliderValues['sales_volume']?? 10) / 100,
      };
      const res = await fetch(getApiUrl() + '/api/tune_weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weights: sliderMap }),
      });
      if (!res.ok) throw new Error('Update failed');
      const data = await res.json();
      if (data.importances) setImportances(data.importances);
      setIsDirty(false);
      setStatusMsg({ text: 'RLHF bias deployed & persisted ✓', type: 'success' });
    } catch (e) {
      setStatusMsg({ text: 'Failed to deploy weights', type: 'error' });
    } finally {
      setIsSaving(false);
      setTimeout(() => setStatusMsg(null), 3000);
    }
  };

  // Restore: zero out RLHF bias → go back to pure LTR model
  const handleRestore = async () => {
    setIsRestoring(true);
    setStatusMsg(null);
    try {
      const res = await fetch(getApiUrl() + '/api/restore_weights', { method: 'POST' });
      if (!res.ok) throw new Error('Restore failed');
      const data = await res.json();
      if (data.importances) {
        setImportances(data.importances);
        const restored: Record<string, number> = {};
        SLIDER_FEATURES.forEach((f) => {
          restored[f] = Math.round((data.importances.effective[f] ?? 0) * 100);
        });
        setSliderValues(restored);
      }
      setIsDirty(false);
      setStatusMsg({ text: 'Restored to LTR model weights ✓', type: 'success' });
    } catch (e) {
      setStatusMsg({ text: 'Failed to restore weights', type: 'error' });
    } finally {
      setIsRestoring(false);
      setTimeout(() => setStatusMsg(null), 3000);
    }
  };

  const handleSliderChange = (feature: string, value: number) => {
    setSliderValues((prev) => ({ ...prev, [feature]: value }));
    setIsDirty(true);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 20, scale: 0.95 }}
      className="fixed top-24 right-6 z-50 w-80 rounded-2xl border border-white/10 shadow-2xl shadow-purple-900/20 overflow-hidden"
      style={{ background: 'rgba(10, 10, 20, 0.92)', backdropFilter: 'blur(24px)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <Brain className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-white tracking-tight">LTR Score Matrix</h3>
            <p className="text-[10px] text-slate-500 font-mono">LightGBM × BGE Embeddings</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors text-slate-400 hover:text-white"
        >
          ✕
        </button>
      </div>

      {/* Model tag */}
      <div className="mx-5 mt-4 mb-3 flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
        <Zap className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
        <p className="text-[10px] text-emerald-300 font-mono">
          Scoring via LambdaRank + BGE-small-en-v1.5
          {hasBias && <span className="text-amber-400 ml-1">· RLHF bias active</span>}
        </p>
      </div>

      {/* Sliders */}
      <div className="px-5 pb-2 max-h-[52vh] overflow-y-auto space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
            <span className="text-xs text-slate-400 ml-2">Loading LTR importances...</span>
          </div>
        ) : (
          SLIDER_FEATURES.map((feature) => {
            const info      = FEATURE_LABELS[feature] ?? { label: feature, icon: '·', description: '' };
            const value     = sliderValues[feature] ?? 0;
            const modelVal  = Math.round((importances?.model_importances?.[feature] ?? 0) * 100);
            const biasVal   = Math.round((importances?.rlhf_bias?.[feature] ?? 0) * 100);
            const isShifted = Math.abs(biasVal) > 1;

            return (
              <div key={feature} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs">{info.icon}</span>
                    <span className="text-[11px] font-semibold text-slate-200">{info.label}</span>
                    {isShifted && (
                      <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded-full ${biasVal > 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                        {biasVal > 0 ? '+' : ''}{biasVal}% bias
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <span className="text-xs font-bold text-purple-300">{value}%</span>
                    {modelVal !== value && (
                      <div className="text-[9px] text-slate-500 font-mono">model: {modelVal}%</div>
                    )}
                  </div>
                </div>

                {/* Track */}
                <div className="relative h-2 w-full rounded-full bg-slate-800 overflow-hidden group">
                  {/* Model baseline marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-slate-600 z-10"
                    style={{ left: `${modelVal}%` }}
                  />
                  {/* Fill */}
                  <motion.div
                    animate={{ width: `${value}%` }}
                    transition={{ type: 'spring', stiffness: 200, damping: 25 }}
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{
                      background: isShifted
                        ? 'linear-gradient(to right, #7c3aed, #f59e0b)'
                        : 'linear-gradient(to right, #7c3aed, #06b6d4)',
                    }}
                  />
                  <input
                    type="range"
                    min={0}
                    max={60}
                    value={value}
                    onChange={(e) => handleSliderChange(feature, parseInt(e.target.value))}
                    title={info.description}
                    className="absolute inset-0 opacity-0 cursor-pointer w-full"
                  />
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Status message */}
      <AnimatePresence>
        {statusMsg && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className={`mx-5 mt-3 px-3 py-2 rounded-xl text-[10px] font-mono flex items-center gap-1.5 ${
              statusMsg.type === 'success'
                ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/20'
                : 'bg-red-500/15 text-red-300 border border-red-500/20'
            }`}
          >
            <CheckCircle2 className="w-3 h-3 flex-shrink-0" />
            {statusMsg.text}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer buttons */}
      <div className="px-5 py-4 mt-2 border-t border-white/5 space-y-2">
        {/* Deploy */}
        <button
          onClick={handleDeploy}
          disabled={isSaving || isLoading || !isDirty}
          className="w-full py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            background: isDirty ? 'linear-gradient(135deg, #7c3aed, #4f46e5)' : 'rgba(124,58,237,0.15)',
            color: isDirty ? '#fff' : '#a78bfa',
            border: '1px solid rgba(124,58,237,0.4)',
          }}
        >
          {isSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
          {isSaving ? 'Deploying...' : 'Deploy RLHF Bias'}
        </button>

        {/* Restore model weights */}
        <button
          onClick={handleRestore}
          disabled={isRestoring || isLoading}
          className="w-full py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            background: 'rgba(15, 15, 30, 0.6)',
            color: '#94a3b8',
            border: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          {isRestoring ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RotateCcw className="w-3.5 h-3.5" />}
          {isRestoring ? 'Restoring...' : 'Restore Model Weights'}
        </button>

        {/* Info */}
        <div className="flex items-start gap-1.5 pt-1">
          <Info className="w-3 h-3 text-slate-600 mt-0.5 flex-shrink-0" />
          <p className="text-[9px] text-slate-600 leading-relaxed">
            Sliders shift the additive RLHF bias on top of the LTR model's learned importances.
            The vertical tick marks show the model's trained baseline. &quot;Restore&quot; zeroes the bias without retraining.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
