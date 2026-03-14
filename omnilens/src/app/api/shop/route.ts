import { NextResponse } from 'next/server';

export const runtime = 'edge'; // Edge runtime is great for streaming

// Mocked 10-item essential checklist for "Skiing trip"
const mockSkiingChecklist = [
  { id: '1', name: 'Thermal Base Layers', category: 'Clothing', essentiality: 0.9, estimatedCost: 60 },
  { id: '2', name: 'Premium Skis', category: 'Gear', essentiality: 1.0, estimatedCost: 700 },
  { id: '3', name: 'Ski Boots', category: 'Gear', essentiality: 1.0, estimatedCost: 350 },
  { id: '4', name: 'Goggles', category: 'Accessories', essentiality: 0.8, estimatedCost: 120 },
  { id: '5', name: 'Waterproof Gloves', category: 'Clothing', essentiality: 0.95, estimatedCost: 80 },
  { id: '6', name: 'Ski Helmet', category: 'Safety', essentiality: 1.0, estimatedCost: 150 },
  { id: '7', name: 'Ski Jacket', category: 'Clothing', essentiality: 0.85, estimatedCost: 300 },
  { id: '8', name: 'Ski Pants', category: 'Clothing', essentiality: 0.85, estimatedCost: 200 },
  { id: '9', name: 'Wool Socks', category: 'Clothing', essentiality: 0.9, estimatedCost: 30 },
  { id: '10', name: 'Action Camera', category: 'Electronics', essentiality: 0.2, estimatedCost: 400 }
];

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export async function POST(req: Request) {
  try {
    const { prompt, budgetStr } = await req.json();
    const budget = parseFloat(budgetStr.replace(/[^0-9.]/g, ''));

    const encoder = new TextEncoder();
    
    const readable = new ReadableStream({
      async start(controller) {
        // Helper to send SSE data
        const sendEvent = (event: string, data: any) => {
          controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        };

        // Phase 1: Semantic Expansion (Instant)
        sendEvent('expansion', {
          message: 'Intent recognized: Activity (Skiing). Extrapolated 10 essentials.',
          items: mockSkiingChecklist.map(item => ({ ...item, status: 'pending', progress: 0 }))
        });

        await sleep(1000); // Simulate NLP processing time
        
        // Phase 3 Preview: Budget Fallback Logic (Greedy Knapsack)
        // Calculate minimum viable cost (sum of items with essentiality >= 0.8)
        let totalCost = 0;
        let selectedItems: typeof mockSkiingChecklist = [];
        let rejectedItems: typeof mockSkiingChecklist = [];
        
        // Sort by essentiality descending, then cost ascending as a tiebreaker
        const sortedItems = [...mockSkiingChecklist].sort(
          (a, b) => b.essentiality - a.essentiality || a.estimatedCost - b.estimatedCost
        );

        for (const item of sortedItems) {
          if (totalCost + item.estimatedCost <= budget) {
            totalCost += item.estimatedCost;
            selectedItems.push(item);
          } else {
            rejectedItems.push(item);
          }
        }

        const isBudgetImpossible = rejectedItems.some(item => item.essentiality > 0.8);
        
        if (isBudgetImpossible) {
            // Trigger Pivot Fallback
            sendEvent('pivot', {
                message: `A full new kit for $${budget} is tough, but I've got you covered. Here are the critical items you can buy under budget, and we suggest renting the rest.`,
                suggestedRentals: rejectedItems.filter(i => i.essentiality > 0.8).map(i => i.name)
            });
        }
        
        // The ones we actually process
        const itemsToProcess = isBudgetImpossible ? selectedItems : mockSkiingChecklist;

        // Phase 2: Concurrent Scraping & ML Evaluation Simulation
        // We simulate parallel processing by wrapping them in async IIFEs that resolve independently
        const processItem = async (item: any) => {
          // Status 1: Scraping
          await sleep(Math.random() * 1500 + 500);
          sendEvent('item_update', { id: item.id, status: 'scraping', statusText: 'Scraping 4 platforms...', progress: 30 });
          
          // Status 2: Analyzing
          await sleep(Math.random() * 2000 + 1000);
          sendEvent('item_update', { id: item.id, status: 'analyzing', statusText: 'Analyzing 400 reviews...', progress: 60 });
          
          // Status 3: Scoring & Complete
          await sleep(Math.random() * 1000 + 500);
          const finalScore = parseFloat((Math.random() * 2 + 8).toFixed(1)); // 8.0 - 10.0
          const finalPrice = parseFloat((item.estimatedCost * (Math.random() * 0.2 + 0.9)).toFixed(2));
          
          sendEvent('item_finish', { 
            id: item.id, 
            status: 'complete',
            score: finalScore,
            platform: 'Amazon/Backcountry',
            finalPrice: finalPrice,
            image: `https://picsum.photos/seed/${item.id}/200/200` // Mock image
          });
        };

        // Fire off all processing concurrently
        await Promise.all(itemsToProcess.map(item => processItem(item)));

        // Signal end of stream
        sendEvent('done', { message: 'All items processed.' });
        controller.close();
      }
    });

    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}
