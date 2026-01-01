import { useCallback, useEffect, useRef } from "react";

export default function useIframeAutoHeight() {
  const frameRef = useRef(null);

  const updateHeight = useCallback(() => {
    const frame = frameRef.current;
    if (!frame) return;
    try {
      const doc = frame.contentDocument || frame.contentWindow?.document;
      if (!doc) return;
      const bodyHeight = doc.body?.scrollHeight ?? 0;
      const docHeight = doc.documentElement?.scrollHeight ?? 0;
      const height = Math.max(bodyHeight, docHeight);
      if (height > 0) {
        frame.style.height = `${height}px`;
      }
    } catch (error) {
      // Ignore cross-origin iframes.
    }
  }, []);

  const handleLoad = useCallback(() => {
    updateHeight();
    setTimeout(updateHeight, 200);
  }, [updateHeight]);

  useEffect(() => {
    window.addEventListener("resize", updateHeight);
    return () => window.removeEventListener("resize", updateHeight);
  }, [updateHeight]);

  return { frameRef, onLoad: handleLoad };
}
