/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx}',
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,jsx,tsx}',
    './public/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        moviqRed: 'var(--moviq-red)',
        moviqRedHover: 'var(--moviq-red-hover)',
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        cardBg: 'var(--card-bg)',
        sidebarBg: 'var(--sidebar-bg)',
      },
      spacing: {
        'screen-90': '90vh',
      },
    },
  },
  plugins: [],
};
