import { useEffect } from "react";
import { WorkspaceLayout } from "./components/WorkspaceLayout";
import { useViewportStore } from "./store/useViewportStore";

export default function App() {
  const initialize = useViewportStore((s) => s.initialize);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  return <WorkspaceLayout />;
}
