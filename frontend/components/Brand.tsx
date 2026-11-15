/** The lens-over-a-trendline mark, inherited from the prototype. */
export function BrandMark() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="7" fill="none" stroke="var(--accent)" strokeWidth="2.2" />
      <path
        d="M5 12h3l2-4 3 7 2-3h2"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M16.5 16.5L21 21" stroke="var(--accent)" strokeWidth="2.4" strokeLinecap="round" />
    </svg>
  );
}
