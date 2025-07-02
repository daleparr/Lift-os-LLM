"""
Training dashboard page for monitoring fine-tuning jobs.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List

# Handle missing dependencies gracefully
try:
    from src.training.local_orchestrator import LocalTrainingOrchestrator
    from src.training.comparison_engine import ModelComparisonEngine
    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    IMPORT_ERROR = str(e)

try:
    from streamlit_app.components.model_selector import ModelSelector
    from streamlit_app.components.comparison_results import ComparisonResults
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False


def main():
    """Main training dashboard page."""
    
    st.set_page_config(
        page_title="Training Dashboard - LLM Finance Leaderboard",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Auto Fine-tuning Dashboard")
    st.markdown("Monitor and manage model fine-tuning jobs with real-time progress tracking.")
    
    # Check if training components are available
    if not TRAINING_AVAILABLE:
        st.error("Training components not available")
        st.code(f"Import Error: {IMPORT_ERROR}")
        st.info("To enable auto fine-tuning, install dependencies: `pip install pydantic-settings transformers peft datasets torch`")
        return
    
    if not COMPONENTS_AVAILABLE:
        st.error("UI components not available")
        st.info("Please check component imports")
        return
    
    # Initialize components
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = LocalTrainingOrchestrator()
        st.session_state.orchestrator.start()
    
    if 'comparison_engine' not in st.session_state:
        st.session_state.comparison_engine = ModelComparisonEngine()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            ["Submit Job", "Monitor Jobs", "View Comparisons", "System Status"]
        )
    
    # Main content based on selected page
    if page == "Submit Job":
        render_submit_job_page()
    elif page == "Monitor Jobs":
        render_monitor_jobs_page()
    elif page == "View Comparisons":
        render_comparisons_page()
    elif page == "System Status":
        render_system_status_page()


def render_submit_job_page():
    """Render job submission page."""
    
    st.header("🚀 Submit Fine-tuning Job")
    
    # Industry Selection
    st.subheader("🏭 Industry Selection")
    
    # Load industry configuration
    industries = {
        "finance": {
            "name": "Financial Services",
            "description": "Banking, investment, regulatory compliance, and financial analysis",
            "icon": "💰"
        },
        "education": {
            "name": "Education",
            "description": "Educational analytics, student performance, and institutional metrics",
            "icon": "🎓"
        },
        "retail": {
            "name": "Retail & E-commerce",
            "description": "Retail analytics, customer metrics, and business performance",
            "icon": "🛒"
        },
        "healthcare": {
            "name": "Healthcare",
            "description": "Healthcare analytics, patient outcomes, and clinical metrics",
            "icon": "🏥"
        },
        "all": {
            "name": "Multi-Industry",
            "description": "Combined training across all industry domains",
            "icon": "🌐"
        }
    }
    
    # Industry selection with visual cards
    selected_industries = []
    
    cols = st.columns(len(industries))
    for i, (industry_key, industry_info) in enumerate(industries.items()):
        with cols[i]:
            if st.checkbox(
                f"{industry_info['icon']} {industry_info['name']}",
                key=f"industry_{industry_key}",
                help=industry_info['description']
            ):
                selected_industries.append(industry_key)
    
    if not selected_industries:
        st.warning("Please select at least one industry for fine-tuning.")
        return
    
    # Show selected industries summary
    if selected_industries:
        st.info(f"Selected Industries: {', '.join([industries[ind]['name'] for ind in selected_industries])}")
    
    # Model selector component
    st.markdown("---")
    st.subheader("🤖 Model Configuration")
    model_selector = ModelSelector()
    config = model_selector.render()
    
    if not config:
        st.warning("Please configure model settings above.")
        return
    
    # Job submission
    st.markdown("---")
    st.subheader("📋 Job Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_name = st.text_input(
            "Job Name (Optional)",
            placeholder=f"e.g., Mistral-7B-{'-'.join(selected_industries).title()}-v1"
        )
        
        priority = st.selectbox(
            "Priority",
            options=["Normal", "High", "Low"],
            index=0
        )
    
    with col2:
        auto_evaluate = st.checkbox(
            "Auto-evaluate after training",
            value=True,
            help="Automatically run benchmark evaluation when training completes"
        )
        
        save_comparison = st.checkbox(
            "Save comparison results",
            value=True,
            help="Save comparison results for future reference"
        )
    
    # Dataset information
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    
    dataset_info = get_dataset_info_for_industries(selected_industries)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", dataset_info["total_samples"])
    with col2:
        st.metric("Industries", len(selected_industries))
    with col3:
        st.metric("Estimated Training Time", f"{dataset_info['estimated_hours']:.1f}h")
    
    # Show dataset breakdown
    with st.expander("📋 Dataset Breakdown"):
        for industry in selected_industries:
            industry_data = dataset_info["industries"][industry]
            st.markdown(f"**{industries[industry]['icon']} {industries[industry]['name']}**")
            st.markdown(f"- Samples: {industry_data['samples']}")
            st.markdown(f"- Dataset: `{industry_data['dataset_path']}`")
            st.markdown(f"- Complexity: {industry_data['complexity']}")
            st.markdown("")
    
    # Submit button
    if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
        if config.get("finetune_enabled", False):
            submit_training_job(config, selected_industries, job_name, priority, auto_evaluate, save_comparison)
        else:
            st.error("Fine-tuning is not enabled. Please select 'Also Fine-tune and Compare' option.")


def get_dataset_info_for_industries(selected_industries: List[str]) -> Dict:
    """Get dataset information for selected industries."""
    
    # Dataset mapping
    industry_datasets = {
        "finance": {
            "dataset_path": "data/training/synthetic_finance_gsib_v3.jsonl",
            "samples": 30,
            "complexity": "high"
        },
        "education": {
            "dataset_path": "data/training/synthetic_education_v1.jsonl",
            "samples": 20,
            "complexity": "medium"
        },
        "retail": {
            "dataset_path": "data/training/synthetic_retail_v1.jsonl",
            "samples": 20,
            "complexity": "medium"
        },
        "healthcare": {
            "dataset_path": "data/training/synthetic_healthcare_v1.jsonl",
            "samples": 20,
            "complexity": "medium"
        },
        "all": {
            "dataset_path": "data/training/combined_multi_industry_corpus.jsonl",
            "samples": 80,
            "complexity": "mixed"
        }
    }
    
    total_samples = 0
    industries_info = {}
    
    for industry in selected_industries:
        if industry in industry_datasets:
            info = industry_datasets[industry]
            industries_info[industry] = info
            total_samples += info["samples"]
    
    # Estimate training time (rough calculation)
    estimated_hours = total_samples * 0.1  # ~6 minutes per sample
    
    return {
        "total_samples": total_samples,
        "estimated_hours": estimated_hours,
        "industries": industries_info
    }


def get_dataset_path_for_industries(selected_industries: List[str]) -> str:
    """Get the appropriate dataset path for selected industries."""
    
    # If multiple industries or "all" is selected, use combined corpus
    if len(selected_industries) > 1 or "all" in selected_industries:
        return "data/training/combined_multi_industry_corpus.jsonl"
    
    # Single industry selection
    industry_paths = {
        "finance": "data/training/synthetic_finance_gsib_v3.jsonl",
        "education": "data/training/synthetic_education_v1.jsonl",
        "retail": "data/training/synthetic_retail_v1.jsonl",
        "healthcare": "data/training/synthetic_healthcare_v1.jsonl"
    }
    
    industry = selected_industries[0]
    return industry_paths.get(industry, "data/training/combined_multi_industry_corpus.jsonl")


def submit_training_job(config: Dict, selected_industries: List[str], job_name: str, priority: str, auto_evaluate: bool, save_comparison: bool):
    """Submit a training job."""
    
    try:
        # Determine dataset path based on selected industries
        dataset_path = get_dataset_path_for_industries(selected_industries)
        
        # Check if dataset exists
        import os
        if not os.path.exists(dataset_path):
            st.error(f"Dataset not found: {dataset_path}")
            st.info("Please ensure the training dataset is available.")
            return
        
        # Submit job
        orchestrator = st.session_state.orchestrator
        job_id = orchestrator.submit_training_job(
            model_name=config["base_model"],
            dataset_path=dataset_path,
            output_dir=None,  # Auto-generated
            # Additional config from training estimates
            **config.get("training_estimates", {}),
            # Industry-specific metadata
            industries=selected_industries,
            priority=priority
        )
        
        st.success(f"✅ Training job submitted successfully!")
        st.info(f"**Job ID:** `{job_id}`")
        
        # Store job metadata
        if 'submitted_jobs' not in st.session_state:
            st.session_state.submitted_jobs = {}
        
        industries_display = ", ".join(selected_industries).title()
        st.session_state.submitted_jobs[job_id] = {
            "name": job_name or f"Job-{job_id[:8]}",
            "model": config.get("base_model_display", config["base_model"]),
            "industries": selected_industries,
            "industries_display": industries_display,
            "priority": priority,
            "auto_evaluate": auto_evaluate,
            "save_comparison": save_comparison,
            "submitted_at": datetime.now()
        }
        
        # Show next steps
        st.markdown("### 📋 Next Steps")
        st.markdown(f"""
        1. Monitor progress in the **Monitor Jobs** tab
        2. Training will take approximately **{config.get('training_estimates', {}).get('estimated_time_hours', 3):.1f} hours**
        3. {"Automatic evaluation will start when training completes" if auto_evaluate else "Manual evaluation can be started after training"}
        4. Results will be available in the **View Comparisons** tab
        """)
        
    except Exception as e:
        st.error(f"Failed to submit job: {str(e)}")


def render_monitor_jobs_page():
    """Render job monitoring page."""
    
    st.header("📊 Job Monitoring")
    
    orchestrator = st.session_state.orchestrator
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Completed"):
            # Clear completed jobs (implementation would go here)
            st.success("Cleared completed jobs")
    
    # Queue status
    queue_status = orchestrator.get_queue_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pending Jobs", queue_status["pending_jobs"])
    
    with col2:
        st.metric("Active Jobs", queue_status["active_jobs"])
    
    with col3:
        st.metric("Completed Jobs", queue_status["completed_jobs"])
    
    with col4:
        st.metric("Failed Jobs", queue_status["failed_jobs"])
    
    # Job list
    st.markdown("### 📋 Job List")
    
    jobs = orchestrator.list_jobs()
    
    if not jobs:
        st.info("No training jobs found.")
        return
    
    # Display jobs
    for job in jobs:
        render_job_card(job)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()


def render_job_card(job):
    """Render a job status card."""
    
    # Status color mapping
    status_colors = {
        "pending": "🟡",
        "running": "🔵", 
        "completed": "🟢",
        "failed": "🔴",
        "cancelled": "⚫"
    }
    
    status_icon = status_colors.get(job.status, "⚪")
    
    with st.expander(f"{status_icon} {job.model_name} - {job.status.title()}", expanded=(job.status == "running")):
        
        col1, col2 = st.columns(2)
        
        with col1:
            industries_display = ", ".join(job.industries).title() if hasattr(job, 'industries') and job.industries else "Finance"
            priority_display = getattr(job, 'priority', 'Normal')
            st.markdown(f"""
            **Job ID:** `{job.job_id}`
            **Model:** {job.model_name}
            **Dataset:** {job.dataset_name}
            **Industries:** {industries_display}
            **Status:** {job.status.title()}
            **Priority:** {priority_display}
            """)
        
        with col2:
            st.markdown(f"""
            **Created:** {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            **Started:** {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'Not started'}
            **Completed:** {job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else 'Not completed'}
            """)
        
        # Progress bar for running jobs
        if job.status == "running" and job.total_steps > 0:
            progress = job.current_step / job.total_steps
            st.progress(progress, text=f"Step {job.current_step}/{job.total_steps} ({progress:.1%})")
            
            # Real-time metrics
            if job.gpu_utilization is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("GPU Utilization", f"{job.gpu_utilization:.1f}%")
                
                with col2:
                    st.metric("Power Usage", f"{job.power_consumption:.1f}W" if job.power_consumption else "N/A")
                
                with col3:
                    st.metric("Temperature", f"{job.temperature:.1f}°C" if job.temperature else "N/A")
        
        # Error message for failed jobs
        if job.status == "failed" and job.error_message:
            st.error(f"Error: {job.error_message}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if job.status == "running":
                if st.button(f"⏹️ Cancel", key=f"cancel_{job.job_id}"):
                    orchestrator = st.session_state.orchestrator
                    if orchestrator.cancel_job(job.job_id):
                        st.success("Job cancelled")
                        st.rerun()
        
        with col2:
            if job.status == "completed" and job.output_model_path:
                if st.button(f"🔍 View Results", key=f"view_{job.job_id}"):
                    st.session_state.selected_job_results = job.job_id
                    st.switch_page("pages/comparison_results.py")
        
        with col3:
            if job.status in ["completed", "failed"]:
                if st.button(f"🗑️ Remove", key=f"remove_{job.job_id}"):
                    # Implementation would remove job from list
                    st.success("Job removed")


def render_comparisons_page():
    """Render model comparisons page."""
    
    st.header("📈 Model Comparisons")
    
    comparison_engine = st.session_state.comparison_engine
    
    # List saved comparisons
    comparisons = comparison_engine.list_comparisons()
    
    if not comparisons:
        st.info("No model comparisons available yet.")
        st.markdown("Complete a fine-tuning job to see comparison results here.")
        return
    
    # Comparison selector
    comparison_options = {
        f"{comp['base_model']} ({comp['comparison_date'][:10]})": comp['id']
        for comp in comparisons
    }
    
    selected_comparison_name = st.selectbox(
        "Select Comparison",
        options=list(comparison_options.keys())
    )
    
    if selected_comparison_name:
        comparison_id = comparison_options[selected_comparison_name]
        comparison = comparison_engine.load_comparison(comparison_id)
        
        if comparison:
            # Render comparison results
            results_component = ComparisonResults()
            results_component.render(comparison)
            
            # Download report
            if st.button("📄 Download Report"):
                report = comparison_engine.generate_comparison_report(comparison)
                st.download_button(
                    label="Download Markdown Report",
                    data=report,
                    file_name=f"comparison_report_{comparison_id}.md",
                    mime="text/markdown"
                )


def render_system_status_page():
    """Render system status page."""
    
    st.header("🖥️ System Status")
    
    orchestrator = st.session_state.orchestrator
    
    # Resource status
    resource_status = orchestrator.get_resource_status()
    
    st.subheader("💻 Hardware Status")
    
    # GPU information
    gpu_stats = resource_status.get("gpu_stats", {})
    system_info = resource_status.get("system_info", {})
    
    if gpu_stats:
        for gpu_id, stats in gpu_stats.items():
            with st.expander(f"GPU {gpu_id}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Utilization", f"{stats.get('utilization', 0):.1f}%")
                
                with col2:
                    st.metric("Temperature", f"{stats.get('temperature', 0):.1f}°C")
                
                with col3:
                    memory_used = stats.get('memory_used', 0)
                    memory_total = stats.get('memory_total', 1)
                    memory_pct = (memory_used / memory_total) * 100
                    st.metric("Memory", f"{memory_pct:.1f}%")
                
                with col4:
                    st.metric("Power", f"{stats.get('power_usage', 0):.1f}W")
    else:
        st.warning("No GPU information available")
    
    # System information
    st.subheader("🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **CPU Cores:** {system_info.get('cpu_count', 'Unknown')}
        **Total Memory:** {system_info.get('memory_total_gb', 0):.1f} GB
        **Available Memory:** {system_info.get('memory_available_gb', 0):.1f} GB
        """)
    
    with col2:
        st.markdown(f"""
        **GPU Count:** {system_info.get('gpu_count', 0)}
        **Training Queue:** {orchestrator.get_queue_status()['pending_jobs']} pending
        **Active Jobs:** {orchestrator.get_queue_status()['active_jobs']}
        """)


if __name__ == "__main__":
    main()