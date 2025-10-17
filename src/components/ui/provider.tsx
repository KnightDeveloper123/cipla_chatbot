"use client"

import { ChakraProvider, createSystem, defaultConfig, defineAnimationStyles, defineConfig } from "@chakra-ui/react"
import {
  ColorModeProvider,
  type ColorModeProviderProps,
} from "./color-mode"

const animationStyles = defineAnimationStyles({
  slideFadeIn: {
    value: {
      animationName: "slide-from-bottom-full, fade-in",
      animationDuration: "0.3s",
    },
  },
  slideFadeOut: {
    value: {
      animationName: "slide-to-bottom-full, fade-out",
      animationDuration: "0.2s",
    },
  },
});

const config = defineConfig({
  theme: {
    tokens: {
      colors: {
        brand: {
          300: { value: "#ff57221f" },
          400: { value: "#d54d2245" },
          500: { value: "#f03824" }, // Default
          700: { value: "#bd200f" },  // Hover
        },
        secondary: {
          300: { value: "#f1f1f1ff" }
        }
      },
    },
    animationStyles,
  },
})

const system = createSystem(defaultConfig, config);

export function Provider(props: ColorModeProviderProps) {
  return (
    <ChakraProvider value={system}>
      <ColorModeProvider {...props} />
    </ChakraProvider>
  )
}
