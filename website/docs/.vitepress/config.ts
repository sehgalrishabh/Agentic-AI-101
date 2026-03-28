import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'Agentic AI 101',
  description: 'Beginner-to-pro guide to building agentic AI systems that do real work.',
  base: '/Agentic-AI-101/',

  themeConfig: {
    logo: '🤖',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Get Started', link: '/introduction' },
      { text: 'GitHub', link: 'https://github.com/sehgalrishabh/agentic-ai-101' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/introduction' },
        ],
      },
      {
        text: 'Part I: The Foundation',
        items: [
          { text: 'Chapter 1: The Agentic Shift', link: '/foundations/01-agentic-shift' },
          { text: 'Chapter 2: Tools & Tech Stack', link: '/foundations/02-tools-tech-stack' },
        ],
      },
      {
        text: "Part II: The Builder's Workshop",
        items: [
          { text: 'Chapter 3: Your First Agent', link: '/builders-workshop/03-first-agent' },
          { text: 'Chapter 4: Memory & Context', link: '/builders-workshop/04-memory-context' },
          { text: 'Chapter 5: Tool Use & Integrations', link: '/builders-workshop/05-tool-use-integrations' },
        ],
      },
      {
        text: 'Part III: Advanced Architectures',
        items: [
          { text: 'Chapter 6: Multi-Agent Systems', link: '/advanced-architectures/06-multi-agent-systems' },
          { text: 'Chapter 7: Autonomous Loops', link: '/advanced-architectures/07-autonomous-loops' },
        ],
      },
      {
        text: 'Part IV: Deployment & Production',
        items: [
          { text: 'Chapter 8: Serving Your Agent', link: '/deployment-production/08-serving-your-agent' },
          { text: 'Chapter 9: Hosting & Cloud', link: '/deployment-production/09-hosting-cloud' },
          { text: 'Chapter 10: Security & Cost', link: '/deployment-production/10-security-cost' },
        ],
      },
      {
        text: 'Part V: Business & Monetization',
        items: [
          { text: 'Chapter 11: Real-World Projects', link: '/business-monetization/11-real-world-projects' },
          { text: 'Chapter 12: The Freelance Path', link: '/business-monetization/12-freelance-path' },
          { text: 'Chapter 13: The SaaS Path', link: '/business-monetization/13-saas-path' },
          { text: 'Chapter 14: Advanced Techniques', link: '/business-monetization/14-advanced-techniques' },
          { text: 'Chapter 15: Future-Proofing', link: '/business-monetization/15-future-proofing' },
        ],
      },
      {
        text: 'Appendix',
        items: [
          { text: 'Cheat Sheets', link: '/appendix/cheat-sheets' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/sehgalrishabh/agentic-ai-101' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 Rishabh Sehgal',
    },

    search: {
      provider: 'local',
    },
  },

  mermaid: {},
}))
