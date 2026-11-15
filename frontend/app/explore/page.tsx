import { ExploreView } from "./ExploreView";

export const metadata = { title: "Explore · HealthLens" };

export default function ExplorePage() {
  return (
    <div className="stack">
      <div>
        <h1 className="pt">Explore the population</h1>
        <p className="muted">
          Share of survey respondents who report each condition, by group. Public, aggregated, no
          personal data.
        </p>
      </div>
      <ExploreView />
    </div>
  );
}
