---
name: Intelligent Retrieval System
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#c2c6d6'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#8c909f'
  outline-variant: '#424754'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e6a'
  primary-container: '#4d8eff'
  on-primary-container: '#00285d'
  inverse-primary: '#005ac2'
  secondary: '#c0c1ff'
  on-secondary: '#1000a9'
  secondary-container: '#3131c0'
  on-secondary-container: '#b0b2ff'
  tertiary: '#ffb786'
  on-tertiary: '#502400'
  tertiary-container: '#df7412'
  on-tertiary-container: '#461f00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#ffdcc6'
  tertiary-fixed-dim: '#ffb786'
  on-tertiary-fixed: '#311400'
  on-tertiary-fixed-variant: '#723600'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-xl:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.01em
  code-sm:
    fontFamily: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px
  2xl: 64px
  gutter: 24px
  sidebar_width: 280px
---

## Brand & Style

The design system is engineered to evoke a sense of "Cognitive Precision." It caters to knowledge workers and researchers who require an environment that feels both academically reliable and technologically advanced. The aesthetic combines the stability of **Corporate Modernism** with the ethereal qualities of **Glassmorphism**.

The visual language focuses on depth and light as metaphors for insight. By using translucent surfaces and concentrated glows, the UI suggests that information is being "brought to light" from deep data stores. The interface remains quiet and unobtrusive to prioritize document readability, while interactive elements pulse with high-tech energy.

## Colors

The palette is rooted in a deep "Midnight Navy" foundation to reduce eye strain during long research sessions. 

- **Primary & Secondary:** The core action color is Electric Blue (#3B82F6), used for primary CTAs and active states. Indigo (#6366F1) is used for secondary accents, such as AI-generated highlights or "thinking" indicators.
- **Backgrounds:** A subtle linear gradient (top-left to bottom-right) from #1E293B to #0F172A creates a sense of infinite depth.
- **Functional Colors:** Success, warning, and error states are desaturated to maintain the dark-mode harmony while ensuring accessibility.

## Typography

This design system utilizes **Inter** for its exceptional legibility and neutral, professional character. 

Hierarchy is established through weight and subtle shifts in tracking. Large display type uses tight tracking and heavy weights to appear authoritative. Body copy is set with generous line heights to ensure long-form document excerpts and AI responses are comfortable to read. Labels and metadata utilize medium weights to distinguish them from standard body text.

## Layout & Spacing

The layout follows a **Fluid Grid** system designed for high-density information. 

- **Structure:** A persistent left-hand sidebar (280px) houses document libraries and chat history. The main content area uses a maximum container width of 1280px for optimal reading length.
- **Rhythm:** An 8px linear scale ensures consistent spacing. Vertical rhythm is critical in the chat interface; use "xl" (40px) spacing between distinct AI/User message blocks and "md" (16px) for internal message elements.
- **Margins:** A standard 24px gutter is used globally to prevent content from touching screen edges.

## Elevation & Depth

Depth is conveyed through **Glassmorphism** and luminosity rather than traditional drop shadows.

- **Surface Tiers:** The base layer is the dark gradient background. The second layer (sidebar, header) uses a solid Navy (#1E293B) to provide structural grounding.
- **Glass Layers:** Foreground cards and the chat input container use a semi-transparent background (alpha 0.4) with a `backdrop-filter: blur(12px)`.
- **The "Inner Glow":** Interactive elements like the primary search input utilize an inner border-glow (0.5px) and a subtle external neon bloom (8px blur, 0.1 opacity) when focused, simulating an active electronic state.

## Shapes

The design system uses a **Rounded** (Level 2) shape language to soften the high-tech aesthetic and make the tool feel approachable. 

- **Standard Elements:** Buttons and input fields use an 8px (0.5rem) radius.
- **Large Containers:** Chat bubbles and document preview cards use a 16px (1rem) radius.
- **Interactive States:** On hover, certain components may slightly expand or increase their corner radius to provide tactile feedback.

## Components

- **Buttons:** Primary buttons feature a subtle blue-to-indigo gradient. Hover states should increase the "bloom" effect behind the button.
- **Input Fields (The AI Command Bar):** The central chat input is the most prominent component. It features a high-blur backdrop, a thin 1px white-alpha border, and a Lucide-style "Sparkles" icon to denote AI capability.
- **Chat Bubbles:** User messages are simple outlined containers; AI responses are glass-morphic cards with a subtle blue left-border accent to denote "System" origin.
- **Source Chips:** Used for RAG citations. These are small, pill-shaped elements with a low-opacity blue background and a hover-trigger for document preview popovers.
- **Sidebars:** Sleek and dark with active states indicated by a vertical bar on the left edge and a subtle text color shift to white.
- **Icons:** Use **Lucide** with a stroke width of 1.5px. Icons should be monochrome (Slate-400) unless active, where they inherit the Primary Blue color.
- **Progress Indicators:** Linear "shimmer" loaders across the top of cards or the main input during RAG retrieval phases.