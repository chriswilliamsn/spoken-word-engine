import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import PricingSection from "@/components/PricingSection";
import Footer from "@/components/Footer";
import TTSInterface from "@/components/TTSInterface";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <HeroSection />
      <TTSInterface />
      <PricingSection />
      <Footer />
    </div>
  );
};

export default Index;
