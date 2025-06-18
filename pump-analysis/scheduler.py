#!/usr/bin/env python3
"""
Pump Analysis Scheduler - Automatic periodic analysis
Runs pump analysis every 12 hours analyzing last 12 hours of data
"""

import os
import time
import logging
import schedule
from datetime import datetime, timedelta
from dotenv import load_dotenv
from main import PumpAnalysisSystem

# Load environment variables from multiple locations
load_dotenv()  # Load from current directory
load_dotenv('../.env')  # Load from parent directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pump_analysis_scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PumpAnalysisScheduler:
    """Scheduler for automatic pump analysis"""
    
    def __init__(self):
        self.analysis_system = None
        self.running = False
        
    def initialize_analysis_system(self):
        """Initialize the pump analysis system"""
        try:
            self.analysis_system = PumpAnalysisSystem()
            logger.info("✅ Pump analysis system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize pump analysis system: {e}")
            return False
    
    def run_periodic_analysis(self):
        """Run periodic analysis (12 hours of data)"""
        logger.info("🚀 Starting periodic pump analysis (12 hours)")
        
        if not self.analysis_system:
            if not self.initialize_analysis_system():
                logger.error("❌ Cannot run analysis - system not initialized")
                return
        
        try:
            # Run analysis for last 12 hours (0.5 days) with limited symbols for performance
            self.analysis_system.run_analysis(days_back=0.5, max_symbols=50)
            logger.info("✅ Periodic analysis completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during periodic analysis: {e}")
            # Reinitialize system on error
            self.analysis_system = None
    
    def run_startup_analysis(self, days_back=7):
        """Run comprehensive startup analysis"""
        logger.info(f"🚀 Starting comprehensive startup analysis ({days_back} days)")
        
        if not self.analysis_system:
            if not self.initialize_analysis_system():
                logger.error("❌ Cannot run startup analysis - system not initialized")
                return
        
        try:
            # Run comprehensive analysis for startup
            self.analysis_system.run_analysis(days_back=days_back, max_symbols=100)
            logger.info("✅ Startup analysis completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during startup analysis: {e}")
            # Reinitialize system on error
            self.analysis_system = None
    
    def start_scheduler(self, run_startup=True, startup_days=7):
        """Start the scheduled pump analysis"""
        logger.info("🔄 Starting Pump Analysis Scheduler")
        
        # Run startup analysis if requested
        if run_startup:
            self.run_startup_analysis(startup_days)
        
        # Schedule periodic analysis every 12 hours
        schedule.every(12).hours.do(self.run_periodic_analysis)
        
        logger.info("⏰ Scheduler configured - analysis every 12 hours")
        logger.info("📊 Next analysis scheduled for: {}".format(
            (datetime.now() + timedelta(hours=12)).strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        self.running = True
        
        # Main scheduler loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Scheduler stopped by user")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Scheduler error: {e}")
            self.running = False
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        logger.info("⏹️ Stopping scheduler...")
        self.running = False

def main():
    """Main function"""
    scheduler = PumpAnalysisScheduler()
    
    # Check for startup options from environment
    run_startup = os.getenv('PUMP_ANALYSIS_STARTUP', 'true').lower() == 'true'
    startup_days = int(os.getenv('PUMP_ANALYSIS_STARTUP_DAYS', '7'))
    
    logger.info(f"🔧 Configuration:")
    logger.info(f"   - Run startup analysis: {run_startup}")
    logger.info(f"   - Startup analysis days: {startup_days}")
    logger.info(f"   - Periodic analysis: every 12 hours")
    
    # Start the scheduler
    scheduler.start_scheduler(run_startup=run_startup, startup_days=startup_days)

if __name__ == "__main__":
    main()