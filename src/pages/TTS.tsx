import TTSInterface from "@/components/TTSInterface";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const TTS = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      <Navbar />
      <main className="pt-20 pb-12">
        <TTSInterface />
      </main>
      <Footer />
    </div>
  );
};

export default TTS;