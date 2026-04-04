'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { getApiUrl } from '@/lib/config';

export type ProductItem = {
    id: string;
    name: string;
    category: string;
    status: 'pending' | 'scraping' | 'analyzing' | 'complete';
    statusText?: string;
    progress: number;
    score?: number;
    platform?: string;
    finalPrice?: number;
    image?: string;
    essentiality?: number;
    external_link: string;
    tags?: string[];
    sentiment?: number;
    reliability?: number;
    brand_score?: number;
    discount_pct?: number;
    sales_volume?: number;
    wait_to_buy?: boolean;
    coupon_applied?: string;
    reddit_sentiment?: string;
    target_query?: string;
    is_explored?: boolean;
};

export type ChatMessage = {
    id: string;
    role: 'user' | 'ai';
    content: string;
    action?: string;
};

export type FallbackData = {
    message: string;
    suggestedRentals: string[];
};

export type ReceiptItem = {
    id: string;
    date: string;
    total: number;
    items: { name: string; price: number }[];
};

type OmniStore = {
    // ── Shopping Session ──────────────────────────────────────────────
    items: ProductItem[];
    allItems: ProductItem[];
    itemsHistory: ProductItem[][];
    fallback: FallbackData | null;
    logs: { id: string; msg: string; time: string }[];
    statusMessage: string;
    isProcessing: boolean;

    setItems: (items: ProductItem[]) => void;
    setAllItems: (items: ProductItem[]) => void;
    pushHistory: () => void;
    popHistory: () => void;
    resetItems: () => void;
    setFallback: (f: FallbackData | null) => void;
    addLog: (msg: string) => void;
    clearLogs: () => void;
    setStatusMessage: (m: string) => void;
    setIsProcessing: (v: boolean) => void;

    // ── Explore Further ───────────────────────────────────────────────
    exploredItems: ProductItem[];
    addExploredItems: (items: ProductItem[]) => void;
    clearExploredItems: () => void;

    // ── Cart ──────────────────────────────────────────────────────────
    cartItems: ProductItem[];
    addToCart: (item: ProductItem) => void;
    removeFromCart: (id: string) => void;
    clearCart: () => void;

    // ── Wishlist ─── full ProductItem objects persisted to localStorage ─
    wishlist: ProductItem[];
    toggleWishlist: (item: ProductItem) => void;
    isWishlisted: (id: string) => boolean;
    clearWishlist: () => void;

    // ── Chat ──────────────────────────────────────────────────────────
    chatMessages: ChatMessage[];
    addChatMessage: (msg: ChatMessage) => void;
    clearChat: () => void;

    // ── Receipts ──────────────────────────────────────────────────────
    receipts: ReceiptItem[];
    addReceipt: (receipt: ReceiptItem) => void;
    removeReceipt: (id: string) => void;
    clearReceipts: () => void;
};

const WELCOME: ChatMessage = {
    id: 'welcome',
    role: 'ai',
    content:
        "👋 Hi! I'm your OmniLens AI assistant.\n\nI've been upgraded with **Deep Search**! I can now:\n• **Deep Dive** — 'show me top 10 mechanical keyboards'\n• Tune rankings — 'I don't care about price'\n• Remove items — 'Remove the cheapest monitor'\n• Replace/Add — 'Replace headphones with speakers'\n• Undo/Reset — 'undo' or 'reset'\n• View results — Head to the Shop tab to see categorized rankings (Trending, Best Deals, etc.)",
};

export const useStore = create<OmniStore>()(
    persist(
        (set, get) => ({
            // ── Shopping Session ──────────────────────────────────────────
            items: [],
            allItems: [],
            itemsHistory: [],
            fallback: null,
            logs: [],
            statusMessage: 'Awaiting prompt sequence...',
            isProcessing: false,

            setItems: (items) => set({ items }),
            setAllItems: (items) => set({ allItems: items }),
            pushHistory: () => {
                const current = get().items;
                set((s) => ({ itemsHistory: [...s.itemsHistory.slice(-19), current] }));
            },
            popHistory: () => {
                const history = get().itemsHistory;
                if (history.length === 0) return;
                const prev = history[history.length - 1];
                set((s) => ({ items: prev, itemsHistory: s.itemsHistory.slice(0, -1) }));
            },
            resetItems: () => {
                set((s) => ({ items: s.allItems, itemsHistory: [], exploredItems: [] }));
            },
            setFallback: (f) => set({ fallback: f }),
            addLog: (msg) =>
                set((s) => {
                    const newLogs = [
                        ...s.logs,
                        { id: Math.random().toString(), msg, time: new Date().toLocaleTimeString([], { hour12: false }) },
                    ];
                    return { logs: newLogs.length > 100 ? newLogs.slice(-100) : newLogs };
                }),
            clearLogs: () => set({ logs: [] }),
            setStatusMessage: (m) => set({ statusMessage: m }),
            setIsProcessing: (v) => set({ isProcessing: v }),

            // ── Explore Further ───────────────────────────────────────────
            exploredItems: [],
            addExploredItems: (newItems) =>
                set((s) => ({ exploredItems: [...s.exploredItems, ...newItems] })),
            clearExploredItems: () => set({ exploredItems: [] }),

            // ── Cart ──────────────────────────────────────────────────────
            cartItems: [],
            addToCart: (item) =>
                set((s) => {
                    const exists = s.cartItems.find((c) => c.id === item.id);
                    if (!exists) {
                        const apiUrl = getApiUrl();
                        fetch(`${apiUrl}/api/rl_feedback`, {
                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ id: item.id, name: item.name, finalPrice: item.finalPrice || 0, platform: item.platform || '', sentiment: item.sentiment || 0, score: item.score || 0, tags: item.tags || [] })
                        }).catch(console.error);
                    }
                    return { cartItems: exists ? s.cartItems : [...s.cartItems, item] };
                }),
            removeFromCart: (id) => set((s) => ({ cartItems: s.cartItems.filter((c) => c.id !== id) })),
            clearCart: () => set({ cartItems: [] }),

            // ── Wishlist ──────────────────────────────────────────────────
            wishlist: [],
            toggleWishlist: (item) =>
                set((s) => {
                    const exists = s.wishlist.some((w) => w.id === item.id);
                    if (!exists) {
                        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
                        fetch(`${apiUrl}/api/rl_feedback`, {
                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ id: item.id, name: item.name, finalPrice: item.finalPrice || 0, platform: item.platform || '', sentiment: item.sentiment || 0, score: item.score || 0, tags: item.tags || [] })
                        }).catch(console.error);
                    }
                    return { wishlist: exists ? s.wishlist.filter((w) => w.id !== item.id) : [...s.wishlist, item] };
                }),
            isWishlisted: (id) => get().wishlist.some((w) => w.id === id),
            clearWishlist: () => set({ wishlist: [] }),

            // ── Chat ──────────────────────────────────────────────────────
            chatMessages: [WELCOME],
            addChatMessage: (msg) => set((s) => ({ chatMessages: [...s.chatMessages, msg] })),
            clearChat: () => set({ chatMessages: [WELCOME] }),

            // ── Receipts ──────────────────────────────────────────────────
            receipts: [],
            addReceipt: (receipt) => set((s) => ({ receipts: [receipt, ...s.receipts] })),
            removeReceipt: (id) => set((s) => ({ receipts: s.receipts.filter(r => r.id !== id) })),
            clearReceipts: () => set({ receipts: [] }),
        }),
        {
            name: 'omnilens-store',
            // Persist cart, wishlist (full objects), and chat across sessions
            partialize: (state) => ({
                cartItems: state.cartItems,
                wishlist: state.wishlist,
                chatMessages: state.chatMessages,
                receipts: state.receipts,
            }),
        }
    )
);
