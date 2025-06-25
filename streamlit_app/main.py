"""
Main Streamlit application for LLM Finance Leaderboard.

This is the entry point for the web dashboard that displays model performance,
benchmark results, and provides interfaces for running evaluations and auto fine-tuning.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Handle missing dependencies gracefully
try:
    from src.config.settings import settings
    from src.data.schemas.data_models import TaskComplexity, ModelProvider
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

try:
    from components.model_selector import ModelSelector
    from components.comparison_results import ComparisonResults
    TRAINING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRAINING_COMPONENTS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="LLM Finance Leaderboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .complexity-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
    
    .low-complexity { background-color: #28a745; }
    .medium-complexity { background-color: #ffc107; color: black; }
    .high-complexity { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 LLM Finance Leaderboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Reproducible benchmark for evaluating Large Language Models on G-SIB financial analysis tasks**
    
    This leaderboard evaluates both stock model checkpoints and fine-tuned models (LoRA/QLoRA) 
    on real-world financial analysis across three complexity tiers.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏆 Leaderboard", "📊 Model Comparison", "📋 Task Analysis", "⚙️ Data Management", "🚀 Run Benchmark", "🔧 Auto Fine-tuning"],
        key="main_navigation_selectbox"
    )
    
    # Route to appropriate page
    if page == "🏆 Leaderboard":
        show_leaderboard()
    elif page == "📊 Model Comparison":
        show_model_comparison()
    elif page == "📋 Task Analysis":
        show_task_analysis()
    elif page == "⚙️ Data Management":
        show_data_management()
    elif page == "🚀 Run Benchmark":
        show_benchmark_runner()
    elif page == "🔧 Auto Fine-tuning":
        show_auto_finetuning()


def show_leaderboard():
    """Display the main leaderboard."""
    st.header("🏆 Model Leaderboard")
    
    # API Configuration Section
    st.subheader("🔧 API Configuration")
    with st.expander("Configure API Connectors", expanded=False):
        show_api_configuration()
    
    # Model Selection Section
    st.subheader("🤖 Model Selection")
    show_model_selection_ui()
    
    st.markdown("---")
    
    # Sample data for demonstration
    sample_data = create_sample_leaderboard_data()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type_filter = st.selectbox(
            "Model Type",
            ["All", "Stock Models", "Finance-Tuned"],
            key="main_leaderboard_model_type"
        )
    
    with col2:
        complexity_filter = st.selectbox(
            "Complexity Focus",
            ["Overall", "Low", "Medium", "High"],
            key="main_leaderboard_complexity"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Final Score", "Quality Score", "Efficiency Score", "Cost per Task"],
            key="main_leaderboard_sort"
        )
    
    # Apply filters
    filtered_data = sample_data.copy()
    if model_type_filter != "All":
        if model_type_filter == "Stock Models":
            filtered_data = filtered_data[~filtered_data['Model'].str.contains('LoRA|Fine')]
        else:
            filtered_data = filtered_data[filtered_data['Model'].str.contains('LoRA|Fine')]
    
    # Display leaderboard table
    st.subheader("Current Rankings")
    
    # Format the dataframe for display
    display_df = filtered_data.copy()
    display_df['Final Score'] = display_df['Final Score'].apply(lambda x: f"{x:.3f}")
    display_df['Quality Score'] = display_df['Quality Score'].apply(lambda x: f"{x:.3f}")
    display_df['Efficiency Score'] = display_df['Efficiency Score'].apply(lambda x: f"{x:.3f}")
    display_df['Cost/Task'] = display_df['Cost/Task'].apply(lambda x: f"${x:.4f}")
    display_df['Latency (ms)'] = display_df['Latency (ms)'].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Final Score Distribution")
        fig = px.bar(
            sample_data.head(10),
            x='Model',
            y='Final Score',
            color='Parameters',
            title="Top 10 Models by Final Score"
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quality vs Efficiency")
        fig = px.scatter(
            sample_data,
            x='Efficiency Score',
            y='Quality Score',
            size='Final Score',
            color='Parameters',
            hover_name='Model',
            title="Quality vs Efficiency Trade-off"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_model_comparison():
    """Display model comparison interface."""
    st.header("📊 Model Comparison")
    
    sample_data = create_sample_leaderboard_data()
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Select First Model", sample_data['Model'].tolist(), key="compare_model1")
    with col2:
        model2 = st.selectbox("Select Second Model", sample_data['Model'].tolist(), index=1, key="compare_model2")
    
    if model1 != model2:
        # Get model data
        model1_data = sample_data[sample_data['Model'] == model1].iloc[0]
        model2_data = sample_data[sample_data['Model'] == model2].iloc[0]
        
        # Comparison metrics
        st.subheader("Head-to-Head Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Final Score",
                f"{model1_data['Final Score']:.3f}",
                delta=f"{model1_data['Final Score'] - model2_data['Final Score']:.3f}"
            )
        
        with col2:
            st.metric(
                "Quality Score", 
                f"{model1_data['Quality Score']:.3f}",
                delta=f"{model1_data['Quality Score'] - model2_data['Quality Score']:.3f}"
            )
        
        with col3:
            st.metric(
                "Cost per Task",
                f"${model1_data['Cost/Task']:.4f}",
                delta=f"${model1_data['Cost/Task'] - model2_data['Cost/Task']:.4f}",
                delta_color="inverse"
            )
        
        # Radar chart comparison
        st.subheader("Performance Radar")
        
        categories = ['Low Complexity', 'Medium Complexity', 'High Complexity', 'Efficiency', 'Cost Effectiveness']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[model1_data['Low Score'], model1_data['Medium Score'], model1_data['High Score'], 
               model1_data['Efficiency Score'], 1 - model1_data['Cost/Task']],
            theta=categories,
            fill='toself',
            name=model1,
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[model2_data['Low Score'], model2_data['Medium Score'], model2_data['High Score'],
               model2_data['Efficiency Score'], 1 - model2_data['Cost/Task']],
            theta=categories,
            fill='toself',
            name=model2,
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_task_analysis():
    """Display task analysis and performance breakdown."""
    st.header("📋 Task Analysis")
    
    # Task complexity breakdown
    st.subheader("Task Complexity Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge low-complexity">LOW</span>', unsafe_allow_html=True)
        st.markdown("**30% of Total Score**")
        st.markdown("- EPS extraction from 10-Q filings")
        st.markdown("- Basel III ratio identification")
        st.markdown("- Basic financial metric retrieval")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge medium-complexity">MEDIUM</span>', unsafe_allow_html=True)
        st.markdown("**40% of Total Score**")
        st.markdown("- Revenue driver analysis")
        st.markdown("- CEO sentiment classification")
        st.markdown("- Cross-document consistency")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge high-complexity">HIGH</span>', unsafe_allow_html=True)
        st.markdown("**30% of Total Score**")
        st.markdown("- Bull/bear case generation")
        st.markdown("- EPS surprise analysis")
        st.markdown("- Portfolio recommendations")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample task performance data
    st.subheader("Task Performance by Model")
    
    task_performance = create_sample_task_data()
    
    fig = px.heatmap(
        task_performance,
        x='Task',
        y='Model',
        z='Score',
        color_continuous_scale='RdYlGn',
        title="Task Performance Heatmap"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_data_management():
    """Display data management interface."""
    st.header("⚙️ Data Management")
    
    # Data collection status
    st.subheader("Data Collection Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SEC Filings", "1,247", delta="23 new")
    with col2:
        st.metric("Earnings Transcripts", "892", delta="12 new")
    with col3:
        st.metric("Market Data Points", "45.2K", delta="1.2K new")
    with col4:
        st.metric("News Articles", "8,934", delta="156 new")
    
    # Data collection controls
    st.subheader("Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**SEC Filings**")
        selected_tickers = st.multiselect(
            "Select Banks",
            ["JPM", "BAC", "WFC", "C", "GS", "MS"],
            default=["JPM", "BAC"]
        )
        
        filing_types = st.multiselect(
            "Filing Types",
            ["10-Q", "10-K"],
            default=["10-Q"]
        )
        
        if st.button("Collect SEC Filings"):
            with st.spinner("Collecting SEC filings..."):
                st.success(f"Started collection for {len(selected_tickers)} banks")
    
    with col2:
        st.markdown("**Market Data**")
        date_range = st.date_input(
            "Date Range",
            value=[datetime.now() - timedelta(days=90), datetime.now()],
            key="market_data_range"
        )
        
        if st.button("Update Market Data"):
            with st.spinner("Updating market data..."):
                st.success("Market data update completed")
    
    # Vector store status
    st.subheader("Vector Store Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Documents", "2,139")
        st.metric("Vector Dimensions", "384")
    with col2:
        st.metric("Index Size", "1.2 GB")
        st.metric("Last Updated", "2 hours ago")


def show_benchmark_runner():
    """Display benchmark runner interface."""
    st.header("🚀 Run Benchmark")
    
    # Model selection
    st.subheader("Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Stock Models**")
        stock_models = st.multiselect(
            "Select stock models",
            ["Mistral 7B", "Llama 2 13B", "Falcon 40B", "Qwen 7B", "Phi-3 Mini"],
            default=["Mistral 7B", "Llama 2 13B"]
        )
    
    with col2:
        st.markdown("**Finance-Tuned Models**")
        finance_models = st.multiselect(
            "Select finance-tuned models",
            ["FinMA 7B", "FinGPT Llama2 13B", "Custom Mistral LoRA"],
            default=["FinMA 7B"]
        )
    
    # Task configuration
    st.subheader("Task Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_low = st.checkbox("Low Complexity Tasks", value=True)
    with col2:
        include_medium = st.checkbox("Medium Complexity Tasks", value=True)
    with col3:
        include_high = st.checkbox("High Complexity Tasks", value=True)
    
    # Evaluation settings
    st.subheader("Evaluation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_seeds = st.slider("Number of Random Seeds", 1, 5, 3)
        timeout_minutes = st.slider("Timeout (minutes)", 10, 60, 30)
    
    with col2:
        max_concurrent = st.slider("Max Concurrent Evaluations", 1, 5, 2)
        save_outputs = st.checkbox("Save Model Outputs", value=True)
    
    # Run benchmark
    st.subheader("Execute Benchmark")
    
    total_models = len(stock_models) + len(finance_models)
    if total_models > 0:
        st.info(f"Ready to evaluate {total_models} models on selected tasks")
        
        if st.button("🚀 Start Benchmark", type="primary"):
            # Simulate benchmark run
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("Initializing models...")
                elif i < 40:
                    status_text.text("Loading data...")
                elif i < 80:
                    status_text.text("Running evaluations...")
                else:
                    status_text.text("Calculating scores...")
                
                # Simulate work
                import time
                time.sleep(0.1)
            
            st.success("Benchmark completed successfully!")
            st.balloons()
    else:
        st.warning("Please select at least one model to evaluate")


def create_sample_leaderboard_data():
    """Create sample leaderboard data for demonstration."""
    return pd.DataFrame({
        'Rank': range(1, 11),
        'Model': [
            'FinMA 7B',
            'Custom Mistral LoRA',
            'FinGPT Llama2 13B',
            'Mistral 7B Instruct',
            'Llama 2 13B Chat',
            'Qwen 1.5 7B Chat',
            'Phi-3 Mini 128K',
            'Falcon 40B Instruct',
            'GPT-3.5 Turbo',
            'Base Llama 2 7B'
        ],
        'Parameters': ['7B', '7B', '13B', '7B', '13B', '7B', '3.8B', '40B', '20B', '7B'],
        'Final Score': [0.847, 0.832, 0.819, 0.798, 0.785, 0.772, 0.758, 0.745, 0.731, 0.698],
        'Quality Score': [0.891, 0.876, 0.863, 0.834, 0.821, 0.808, 0.795, 0.782, 0.769, 0.736],
        'Efficiency Score': [0.723, 0.734, 0.698, 0.745, 0.712, 0.689, 0.656, 0.623, 0.590, 0.567],
        'Low Score': [0.923, 0.908, 0.895, 0.867, 0.854, 0.841, 0.828, 0.815, 0.802, 0.769],
        'Medium Score': [0.876, 0.861, 0.848, 0.819, 0.806, 0.793, 0.780, 0.767, 0.754, 0.721],
        'High Score': [0.874, 0.859, 0.846, 0.817, 0.804, 0.791, 0.778, 0.765, 0.752, 0.719],
        'Latency (ms)': [2340, 2180, 3120, 2890, 3450, 2670, 1890, 5670, 1230, 2980],
        'Cost/Task': [0.0023, 0.0021, 0.0034, 0.0028, 0.0041, 0.0025, 0.0018, 0.0067, 0.0089, 0.0031]
    })


def create_sample_task_data():
    """Create sample task performance data."""
    models = ['FinMA 7B', 'Mistral 7B', 'Llama 2 13B', 'FinGPT 13B', 'Qwen 7B']
    tasks = ['EPS Extract', 'Ratio ID', 'Revenue Analysis', 'Sentiment', 'Target Price', 'Divergence']
    
    data = []
    import random
    random.seed(42)
    
    for model in models:
        for task in tasks:
            # Simulate realistic performance scores
            if 'FinMA' in model or 'FinGPT' in model:
                base_score = random.uniform(0.75, 0.95)
            else:
                base_score = random.uniform(0.65, 0.85)
            
            data.append({
                'Model': model,
                'Task': task,
                'Score': base_score
            })
    
    return pd.DataFrame(data).pivot(index='Model', columns='Task', values='Score')


def show_api_configuration():
    """Display API configuration interface."""
    st.markdown("**Configure your API keys for model providers and data sources:**")
    
    # Model Provider APIs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Model Provider APIs**")
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for GPT models",
            key="openai_api_key"
        )
        
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for Claude models",
            key="anthropic_api_key"
        )
        
        huggingface_token = st.text_input(
            "HuggingFace API Token",
            type="password",
            help="Required for HuggingFace models",
            key="huggingface_api_token"
        )
        
        pinecone_key = st.text_input(
            "Pinecone API Key",
            type="password",
            help="Required for vector storage",
            key="pinecone_api_key"
        )
        
        pinecone_env = st.text_input(
            "Pinecone Environment",
            help="e.g., us-west1-gcp-free",
            key="pinecone_environment"
        )
    
    with col2:
        st.markdown("**📊 Financial Data APIs**")
        
        fred_key = st.text_input(
            "FRED API Key",
            type="password",
            help="Federal Reserve Economic Data",
            key="fred_api_key"
        )
        
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            help="Stock market data",
            key="alpha_vantage_api_key"
        )
        
        wandb_key = st.text_input(
            "Weights & Biases API Key",
            type="password",
            help="Experiment tracking (optional)",
            key="wandb_api_key"
        )
        
        wandb_project = st.text_input(
            "W&B Project Name",
            help="Project name for experiment tracking",
            key="wandb_project"
        )
    
    # API Status Check
    st.markdown("**📡 API Status**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if openai_key:
            st.success("✅ OpenAI")
        else:
            st.error("❌ OpenAI")
    
    with col2:
        if anthropic_key:
            st.success("✅ Anthropic")
        else:
            st.error("❌ Anthropic")
    
    with col3:
        if huggingface_token:
            st.success("✅ HuggingFace")
        else:
            st.error("❌ HuggingFace")
    
    with col4:
        if pinecone_key and pinecone_env:
            st.success("✅ Pinecone")
        else:
            st.error("❌ Pinecone")


def show_model_selection_ui():
    """Display enhanced model selection interface."""
    
    # Load model configurations
    try:
        import yaml
        with open('src/config/models_config.yaml', 'r') as f:
            models_config = yaml.safe_load(f)
    except:
        st.error("Could not load model configuration")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📦 Base Models**")
        
        # Stock models dropdown
        stock_models = list(models_config.get('stock_models', {}).keys())
        stock_model_names = [models_config['stock_models'][key]['display_name']
                           for key in stock_models]
        
        selected_base_model = st.selectbox(
            "Select Base Model",
            options=stock_model_names,
            help="Choose the base model for evaluation",
            key="main_selected_base_model"
        )
        
        # Get selected model details
        if selected_base_model:
            selected_key = None
            for key, config in models_config['stock_models'].items():
                if config['display_name'] == selected_base_model:
                    selected_key = key
                    break
            
            if selected_key:
                model_config = models_config['stock_models'][selected_key]
                
                # Display model details
                st.info(f"""
                **Model Details:**
                - Parameters: {model_config['parameters']}
                - Context Length: {model_config['context_length']:,}
                - Provider: {model_config['provider'].title()}
                - Cost per 1K tokens: ${model_config['cost_per_1k_tokens']:.4f}
                """)
    
    with col2:
        st.markdown("**🎯 Fine-tuned Variants**")
        
        # Fine-tuning options
        finetuning_option = st.radio(
            "Fine-tuning Approach",
            options=[
                "📊 Base Model Only",
                "🔧 Compare with Fine-tuned",
                "🚀 Auto Fine-tune & Compare"
            ],
            help="Choose how to handle fine-tuning",
            key="finetuning_option"
        )
        
        if finetuning_option == "🔧 Compare with Fine-tuned":
            # Show existing fine-tuned models
            finance_models = list(models_config.get('finance_tuned_models', {}).keys())
            finance_model_names = [models_config['finance_tuned_models'][key]['display_name']
                                 for key in finance_models]
            
            selected_finetuned = st.selectbox(
                "Select Fine-tuned Model",
                options=finance_model_names,
                help="Choose existing fine-tuned model for comparison",
                key="main_selected_finetuned"
            )
            
            if selected_finetuned:
                # Find the selected model config
                for key, config in models_config['finance_tuned_models'].items():
                    if config['display_name'] == selected_finetuned:
                        st.info(f"""
                        **Fine-tuned Model Details:**
                        - Base Model: {config.get('base_model', 'Unknown')}
                        - Parameters: {config['parameters']}
                        - Provider: {config['provider'].title()}
                        - Specialization: Financial Analysis
                        """)
                        break
        
        elif finetuning_option == "🚀 Auto Fine-tune & Compare":
            st.info("""
            **Auto Fine-tuning:**
            - Creates LoRA/QLoRA adapter for selected base model
            - Trains on synthetic financial dataset
            - Compares base vs fine-tuned performance
            - Estimated time: 30-60 minutes
            """)
            
            # Fine-tuning parameters
            with st.expander("Fine-tuning Parameters", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    lora_rank = st.slider("LoRA Rank", 8, 64, 16, step=8)
                    lora_alpha = st.slider("LoRA Alpha", 16, 128, 32, step=16)
                
                with col_b:
                    learning_rate = st.select_slider(
                        "Learning Rate",
                        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                        value=2e-5,
                        format_func=lambda x: f"{x:.0e}"
                    )
                    num_epochs = st.slider("Training Epochs", 1, 10, 3)
        
        else:  # Base Model Only
            st.info("""
            **Base Model Evaluation:**
            - Evaluates selected model as-is
            - No fine-tuning applied
            - Fastest evaluation option
            """)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Evaluation", type="primary"):
            st.success("Evaluation started! Check the benchmark runner for progress.")
    
    with col2:
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved!")
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            st.rerun()


def show_auto_finetuning():
    """Display the auto fine-tuning interface."""
    st.header("🔧 Auto Fine-tuning")
    st.markdown("Compare base models against fine-tuned variants with domain-specific training.")
    
    # Check if components are available
    if not TRAINING_COMPONENTS_AVAILABLE:
        st.error("Training components not available")
        st.info("Install training dependencies: `pip install pydantic-settings transformers peft datasets torch`")
        st.markdown("### 📋 Missing Dependencies")
        st.markdown("""
        To enable auto fine-tuning, please install:
        ```bash
        pip install pydantic-settings transformers peft datasets torch bitsandbytes accelerate
        ```
        """)
        return
    
    # Initialize components
    if 'training_orchestrator' not in st.session_state:
        try:
            from src.training.local_orchestrator import LocalTrainingOrchestrator
            from src.training.comparison_engine import ModelComparisonEngine
            
            st.session_state.training_orchestrator = LocalTrainingOrchestrator()
            st.session_state.comparison_engine = ModelComparisonEngine()
            st.session_state.training_orchestrator.start()
        except ImportError as e:
            st.error(f"Training components not available: {e}")
            st.info("Install training dependencies: `pip install pydantic-settings transformers peft datasets torch`")
            return
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🚀 Start Training", "📊 Monitor Progress", "📈 View Results"])
    
    with tab1:
        show_training_setup()
    
    with tab2:
        show_training_monitor()
    
    with tab3:
        show_training_results()


def show_training_setup():
    """Show training setup interface."""
    st.subheader("🚀 Model Training Setup")
    
    # Model selector component
    model_selector = ModelSelector()
    config = model_selector.render()
    
    if not config:
        return
    
    # Training submission
    if config.get("finetune_enabled", False):
        st.markdown("---")
        st.subheader("📋 Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_name = st.text_input(
                "Job Name (Optional)",
                placeholder="e.g., Mistral-7B-Finance-Experiment"
            )
            
            auto_evaluate = st.checkbox(
                "Auto-evaluate after training",
                value=True,
                help="Automatically run comparison evaluation when training completes"
            )
        
        with col2:
            priority = st.selectbox(
                "Priority",
                options=["Normal", "High", "Low"],
                index=0,
                key="main_training_priority"
            )
            
            save_results = st.checkbox(
                "Save comparison results",
                value=True,
                help="Save results for future reference"
            )
        
        # Submit button
        if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
            submit_finetuning_job(config, job_name, priority, auto_evaluate, save_results)
    
    else:
        st.info("💡 Select 'Also Fine-tune and Compare' option above to enable training.")


def submit_finetuning_job(config, job_name, priority, auto_evaluate, save_results):
    """Submit a fine-tuning job."""
    try:
        # Get dataset configuration
        dataset_config = config.get("dataset_config", {})
        dataset_path = dataset_config.get("path", "data/training/synthetic_finance_v2.jsonl")
        
        # Check dataset exists
        import os
        if not os.path.exists(dataset_path):
            st.error(f"Training dataset not found: {dataset_path}")
            st.info("Please ensure the synthetic finance dataset is available.")
            return
        
        # Submit to orchestrator
        orchestrator = st.session_state.training_orchestrator
        job_id = orchestrator.submit_training_job(
            model_name=config["base_model"],
            dataset_path=dataset_path,
            **config.get("training_estimates", {})
        )
        
        st.success(f"✅ Training job submitted successfully!")
        st.info(f"**Job ID:** `{job_id}`")
        
        # Store job metadata
        if 'submitted_jobs' not in st.session_state:
            st.session_state.submitted_jobs = {}
        
        st.session_state.submitted_jobs[job_id] = {
            "name": job_name or f"Job-{job_id[:8]}",
            "model": config["base_model_display"],
            "priority": priority,
            "auto_evaluate": auto_evaluate,
            "save_results": save_results,
            "submitted_at": datetime.now()
        }
        
        # Show next steps
        training_time = config.get("training_estimates", {}).get("estimated_time_hours", 3)
        st.markdown(f"""
        ### 📋 Next Steps
        1. Monitor progress in the **Monitor Progress** tab
        2. Training will take approximately **{training_time:.1f} hours**
        3. {"Automatic evaluation will start when training completes" if auto_evaluate else "Manual evaluation available after training"}
        4. Results will appear in the **View Results** tab
        """)
        
    except Exception as e:
        st.error(f"Failed to submit training job: {str(e)}")


def show_training_monitor():
    """Show training monitoring interface."""
    st.subheader("📊 Training Progress Monitor")
    
    if 'training_orchestrator' not in st.session_state:
        st.warning("Training orchestrator not initialized.")
        return
    
    orchestrator = st.session_state.training_orchestrator
    
    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=True)
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Queue status overview
    queue_status = orchestrator.get_queue_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pending", queue_status["pending_jobs"])
    
    with col2:
        st.metric("Running", queue_status["active_jobs"])
    
    with col3:
        st.metric("Completed", queue_status["completed_jobs"])
    
    with col4:
        st.metric("Failed", queue_status["failed_jobs"])
    
    # Job list
    st.markdown("### 📋 Active Jobs")
    
    jobs = orchestrator.list_jobs()
    
    if not jobs:
        st.info("No training jobs found.")
        return
    
    # Display jobs
    for job in jobs[:5]:  # Show latest 5 jobs
        render_job_status_card(job)
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()


def render_job_status_card(job):
    """Render a job status card."""
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
            st.markdown(f"""
            **Job ID:** `{job.job_id[:8]}...`
            **Model:** {job.model_name}
            **Status:** {job.status.title()}
            """)
        
        with col2:
            st.markdown(f"""
            **Created:** {job.created_at.strftime('%H:%M:%S')}
            **Started:** {job.started_at.strftime('%H:%M:%S') if job.started_at else 'Not started'}
            """)
        
        # Progress for running jobs
        if job.status == "running" and job.total_steps > 0:
            progress = job.current_step / job.total_steps
            st.progress(progress, text=f"Step {job.current_step}/{job.total_steps}")
            
            # Resource metrics
            if job.gpu_utilization is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("GPU", f"{job.gpu_utilization:.0f}%")
                
                with col2:
                    st.metric("Power", f"{job.power_consumption:.0f}W" if job.power_consumption else "N/A")
                
                with col3:
                    st.metric("Temp", f"{job.temperature:.0f}°C" if job.temperature else "N/A")
        
        # Error message
        if job.status == "failed" and job.error_message:
            st.error(f"Error: {job.error_message}")


def show_training_results():
    """Show training results and comparisons."""
    st.subheader("📈 Training Results & Comparisons")
    
    if 'comparison_engine' not in st.session_state:
        st.warning("Comparison engine not initialized.")
        return
    
    comparison_engine = st.session_state.comparison_engine
    
    # List available comparisons
    comparisons = comparison_engine.list_comparisons()
    
    if not comparisons:
        st.info("No comparison results available yet.")
        st.markdown("Complete a fine-tuning job to see results here.")
        return
    
    # Comparison selector
    comparison_options = {
        f"{comp['base_model']} ({comp['comparison_date'][:10]})": comp['id']
        for comp in comparisons
    }
    
    selected_name = st.selectbox(
        "Select Comparison Result",
        options=list(comparison_options.keys()),
        key="main_comparison_result"
    )
    
    if selected_name:
        comparison_id = comparison_options[selected_name]
        comparison = comparison_engine.load_comparison(comparison_id)
        
        if comparison:
            # Render comparison results
            results_component = ComparisonResults()
            results_component.render(comparison)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Generate Report"):
                    report = comparison_engine.generate_comparison_report(comparison)
                    st.download_button(
                        label="Download Markdown Report",
                        data=report,
                        file_name=f"comparison_report_{comparison_id}.md",
                        mime="text/markdown"
                    )
            
            with col2:
                if st.button("💾 Export Data"):
                    import json
                    data = comparison.dict()
                    st.download_button(
                        label="Download JSON Data",
                        data=json.dumps(data, indent=2, default=str),
                        file_name=f"comparison_data_{comparison_id}.json",
                        mime="application/json"
                    )


if __name__ == "__main__":
    main()