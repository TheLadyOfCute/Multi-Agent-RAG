---
name: High-End Intelligence
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
  secondary: '#d0bcff'
  on-secondary: '#3c0091'
  secondary-container: '#571bc1'
  on-secondary-container: '#c4abff'
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
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#d0bcff'
  on-secondary-fixed: '#23005c'
  on-secondary-fixed-variant: '#5516be'
  tertiary-fixed: '#ffdcc6'
  tertiary-fixed-dim: '#ffb786'
  on-tertiary-fixed: '#311400'
  on-tertiary-fixed-variant: '#723600'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-lg:
    fontFamily: Space Grotesk
    fontSize: 3.5rem
    fontWeight: '700'
    lineHeight: '1.1'
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: '1.3'
  body-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: '1.5'
  label-sm:
    fontFamily: Manrope
    fontSize: 0.6875rem
    fontWeight: '500'
    lineHeight: '1.2'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
---

# Design System Specification: High-End Intelligence

## 1. Overview & Creative North Star
**Creative North Star: The Luminous Engine (Dark Edition)**
This design system moves beyond the "SaaS-in-a-box" aesthetic. Instead of a rigid grid of outlines, we treat the interface as an architectural space of depth and focused light. We are building a "Luminous Engine"—an environment that feels immersive, intelligent, and editorial. 

The system breaks the "template" look through **intentional asymmetry**, high-contrast typography scales, and **tonal layering**. We favor breathing room and typographic authority over dense, bordered modules. The goal is to make the user feel they are interacting with a sophisticated instrument, utilizing a deep, professional canvas for maximum focus.

---

## 2. Colors & Surface Philosophy
The palette is built on a foundation of deep obsidian and midnight tones, punctuated by high-energy electric pulses of blue and violet.

### Surface Hierarchy & Nesting
To achieve a premium "Editorial" feel in a dark environment, we follow the **Tonal Layering Principle**. Instead of using borders to separate sections, we use the `surface_container` tiers to define depth through slight increases in luminosity.
*   **Base Layer:** `surface` – The deep, midnight (#0F172A) canvas.
*   **Structural Sections:** `surface_container_low` – Soft, subtle dark regions for background definition.
*   **Primary Modules:** `surface_container` – Standard card backgrounds.
*   **Interactive/Elevated:** `surface_container_high` – Hover states or active modules that "rise" toward the light.

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders for sectioning or card definition. Boundaries must be defined solely through background color shifts or subtle tonal transitions. A `surface_container_low` section sitting on a `surface` background creates a clear, sophisticated boundary without the visual "noise" of a line.

### The "Glass & Gradient" Rule
For floating elements (modals, popovers, navigation rails), use **Glassmorphism**:
*   **Background:** Use `surface_variant` at 60-80% opacity.
*   **Effect:** Apply a `backdrop-blur` of 12px to 20px.
*   **Signature Texture:** Use a subtle linear gradient for primary CTAs, transitioning from `primary` (#3B82F6) to a slightly darker tint at a 135-degree angle. This provides a "shimmer" that flat colors lack.

---

## 3. Typography
Our typography is a dialogue between technical precision and editorial character, optimized for dark-mode legibility.

*   **Display & Headlines (Space Grotesk):** This is our "Cutting-Edge" voice. The geometric quirks of Space Grotesk should be used for large titles and data hero numbers. It feels intentional and high-end.
*   **Body (Inter):** The "Reliable" voice. Inter is used for technical data, chat logs, and long-form descriptions. It ensures maximum readability in low-light environments.
*   **Labels (Manrope):** The "Professional" voice. Manrope’s modern structure is perfect for small metadata, button labels, and system status tags.

### Hierarchy Guidelines
*   **display-lg (3.5rem):** Reserved for core value propositions or landing hero states.
*   **headline-sm (1.5rem):** Standard for page titles and large card headers.
*   **body-md (0.875rem):** The workhorse for all data and chat interactions.
*   **label-sm (0.6875rem):** Used for "AI Thinking" indicators and secondary metadata.

---

## 4. Elevation & Depth
We reject traditional heavy drop shadows in favor of **Tonal Stacking and Light Catchers**.

*   **Tonal Stacking:** Place a `surface_container_highest` card on a `surface_container_low` section to create a crisp, "elevated" effect.
*   **Ambient Shadows:** When an element must "float" (e.g., a context menu), use a shadow with a blur of 40px, 0% spread, and a low opacity (approx 10-15%) using the `neutral` palette. 
*   **The "Ghost Border" Fallback:** If accessibility requires a border, use the `outline_variant` token at **15% opacity**. It should appear as a subtle edge catch, not a hard line.

---

## 5. Components

### AI Interaction Blocks
*   **User Message:** `surface_container_high` with `md` (roundedness: 2) corners.
*   **Assistant Message:** `surface_container_low` (deep midnight) with a `primary` accent on the left edge (2px width).
*   **"AI Thinking" Block:** Use a soft tint of `secondary` (#8B5CF6) at 10% opacity with a subtle pulsing animation. Typography should be `label-md`.

### Buttons
*   **Primary:** Gradient of `primary` to a slightly darker variant. Text in `on_primary`. Roundedness: `full` (pill).
*   **Secondary:** Ghost style. No background, `outline_variant` (40% opacity) border. Text in `primary`.
*   **Tertiary:** `surface_container_highest` background. Text in `on_surface_variant`.

### Cards & Data Lists
*   **Prohibition:** Never use divider lines between list items. 
*   **Separation:** Use 16px of vertical space or alternating `surface_container` and `surface_container_low` backgrounds for "Zebra" styling.
*   **Roundedness:** Standard cards use `lg` (roundedness: 2); interactive chips use `full`.

### Input Fields
*   **State:** Unfocused inputs should be `surface_container_lowest` (the darkest base). 
*   **Focus State:** Shift to `surface_container_high` with a 1px "Ghost Border" of `primary` at 50% opacity.

---

## 6. Do's and Don'ts

### Do
*   **Do** embrace negative space. High-end dark design requires room to breathe to avoid visual density.
*   **Do** use `secondary` (#8B5CF6) for insights to provide a sophisticated contrast to the primary blue.
*   **Do** use `display-md` for large data points to make statistics feel like glowing high-end typography.

### Don't
*   **Don't** use pure black (#000000) for backgrounds. Use `neutral` (#0F172A) to maintain depth and color harmony.
*   **Don't** use 1px solid borders to "organize" the layout. Use background shifts.
*   **Don't** use standard "drop shadows." In dark mode, light surfaces indicate elevation better than dark shadows.
*   **Don't** mix more than two font families in a single component. Stick to the Display/Body pairing.