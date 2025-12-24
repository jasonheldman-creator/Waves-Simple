import React from "react";
import WaveCard from "../WaveCard";

const WaveCards = () => {
  return (
    <div>
      <WaveCard
        title="Large Cap Growth"
        description=
          "This wave primarily emphasizes stocks with a market cap in the $10 billion+ range. Most are in the growth sectors."
      />
      <WaveCard
        title="Small–Mid Cap Blend"
        description=
          "Focuses on potential gainers in the $1–10 billion range. Looks at both growth and value characteristics."
      />
      <WaveCard
        title="Non-U.S. International"
        description=
          "Diversifies assets outside the U.S. by adding high-performing international stocks. Average yield: 40–60%."
      />
    </div>
  );
};

export default WaveCards;