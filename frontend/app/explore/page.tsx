import { ExploreView } from "./ExploreView";

export const metadata = { title: "Explore · HealthLens" };

export default function ExplorePage() {
  return (
    <div className="stack">
      <div>
        <h1 className="pt">Explore the population</h1>
        <p className="muted">Aggregated survey responses. No personal data.</p>
      </div>
      <ExploreView />
    </div>
  );
}
