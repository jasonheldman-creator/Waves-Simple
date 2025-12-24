import React from 'react';

type WaveCardsProps = {
  title: string;
  content: string;
};

const WaveCards: React.FC<WaveCardsProps> = ({ title, content }) => {
  return (
    <div className="wave-card">
      <h3>{title}</h3>
      <p>
        {content.split('\n').map((line, index) => (
          <React.Fragment key={index}>
            {line}
            <br />
          </React.Fragment>
        ))}
      </p>
      <p>Example usage: \`Code snippet here\`</p>
    </div>
  );
};

export default WaveCards;