'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Aperture, Home, Heart, ShoppingCart, MessageSquare, Receipt } from 'lucide-react';
import { useStore } from '@/store/useStore';
import ThemeToggle from './ThemeToggle';

export default function NavBar() {
    const pathname = usePathname();
    const cartItems = useStore((s) => s.cartItems);
    const wishlist = useStore((s) => s.wishlist);
    const receipts = useStore((s) => s.receipts);

    const links = [
        { href: '/', label: 'Shop', icon: Home },
        { href: '/wishlist', label: 'Wishlist', icon: Heart, badge: wishlist.filter(w => w.status === 'complete').length },
        { href: '/cart', label: 'Cart', icon: ShoppingCart },
        { href: '/receipts', label: 'Receipts', icon: Receipt },
        { href: '/chat', label: 'Assistant', icon: MessageSquare },
    ];

    return (
        <>
            {/* Desktop top nav */}
            <nav className="hidden md:flex items-center justify-between px-8 py-3 border-b border-card-border bg-nav-bg backdrop-blur-2xl sticky top-0 z-50 shadow-sm">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2.5 group">
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center shadow-lg group-hover:shadow-purple-500/50 transition-all">
                        <Aperture className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-black text-lg tracking-tight text-slate-800 dark:text-transparent dark:bg-clip-text dark:bg-gradient-to-r dark:from-white dark:to-slate-400">
                        OMNI<span className="text-purple-600 dark:text-purple-400">LENS</span>
                    </span>
                    <span className="text-[10px] font-mono tracking-[0.2em] text-slate-400 dark:text-slate-600 uppercase self-end mb-0.5">Pro</span>
                </Link>

                {/* Links */}
                <div className="flex items-center gap-1">
                    {links.map(({ href, label, icon: Icon, badge }) => {
                        const active = pathname === href;
                        return (
                            <Link key={href} href={href}
                                className={`relative flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${active
                                    ? 'bg-purple-100 dark:bg-purple-500/20 text-purple-600 dark:text-purple-300 shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-white/5'
                                    }`}>
                                <Icon className="w-4 h-4" />
                                {label}
                                {!!badge && badge > 0 && (
                                    <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-gradient-to-br from-purple-500 to-cyan-500 text-white text-[9px] font-black flex items-center justify-center shadow-md">
                                        {badge > 9 ? '9+' : badge}
                                    </span>
                                )}
                            </Link>
                        );
                    })}
                </div>

                {/* Status dot and Theme Toggle */}
                <div className="flex items-center gap-4">
                    <ThemeToggle />
                    <div className="flex items-center gap-2">
                        <div className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
                        </div>
                        <span className="text-[10px] font-mono tracking-widest text-slate-500 dark:text-slate-600 uppercase">Online</span>
                    </div>
                </div>
            </nav>

            {/* Mobile bottom nav */}
            <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 flex items-center justify-around px-2 py-2 bg-nav-bg backdrop-blur-2xl border-t border-card-border shadow-lg">
                <div className="flex flex-col items-center">
                    <ThemeToggle />
                </div>
                {links.map(({ href, label, icon: Icon, badge }) => {
                    const active = pathname === href;
                    return (
                        <Link key={href} href={href}
                            className={`relative flex flex-col items-center gap-1 px-4 py-1.5 rounded-xl transition-all ${active ? 'text-purple-400' : 'text-slate-600 hover:text-slate-400'
                                }`}>
                            <Icon className="w-5 h-5" />
                            <span className="text-[9px] font-mono tracking-widest uppercase">{label}</span>
                            {!!badge && badge > 0 && (
                                <span className="absolute top-0 right-1 w-3.5 h-3.5 rounded-full bg-purple-500 text-white text-[8px] font-black flex items-center justify-center">
                                    {badge > 9 ? '9+' : badge}
                                </span>
                            )}
                        </Link>
                    );
                })}
            </nav>
        </>
    );
}
