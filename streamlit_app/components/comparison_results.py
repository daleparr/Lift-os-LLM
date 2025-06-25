"""
Comparison results component for displaying base vs fine-tuned model performance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from src.data.schemas.data_models import ModelComparison


class ComparisonResults:
    """Component for displaying model comparison results."""
    
    def __init__(self):
        """Initialize comparison results component."""
        pass
    
    def render(self, comparison: ModelComparison) -> None:
        """Render the comparison results."""
        
        st.subheader("📊 Model Performance Comparison")
        
        # Overview metrics
        self._render_overview(comparison)
        
        # Detailed comparison table
        self._render_comparison_table(comparison)
        
        # Performance visualization
        self._render_performance_charts(comparison)
        
        # Training metrics
        self._render_training_metrics(comparison)
        
        # ROI analysis
        self._render_roi_analysis(comparison)
    
    def _render_overview(self, comparison: ModelComparison) -> None:
        """Render overview metrics."""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Improvement",
                f"{comparison.overall_improvement:.1%}",
                delta=f"{comparison.overall_improvement:.1%}",
                help="Average improvement across all tasks"
            )
        
        with col2:
            training_hours = comparison.training_time_hours
            st.metric(
                "Training Time",
                f"{training_hours:.1f}h",
                help="Total time spent fine-tuning"
            )
        
        with col3:
            power_kwh = comparison.power_consumption_kwh
            st.metric(
                "Power Used",
                f"{power_kwh:.1f} kWh",
                help="Total power consumption during training"
            )
        
        with col4:
            # Calculate ROI (improvement per hour)
            roi_per_hour = comparison.overall_improvement / training_hours if training_hours > 0 else 0
            st.metric(
                "ROI per Hour",
                f"{roi_per_hour:.2%}",
                help="Performance improvement per training hour"
            )
    
    def _render_comparison_table(self, comparison: ModelComparison) -> None:
        """Render detailed comparison table."""
        
        st.markdown("### 📋 Detailed Results")
        
        # Prepare data for table
        table_data = []
        
        for task_name in comparison.base_scores:
            if task_name in comparison.finetuned_scores and task_name in comparison.improvements:
                base_score = comparison.base_scores[task_name]
                finetuned_score = comparison.finetuned_scores[task_name]
                improvement = comparison.improvements[task_name]
                
                # Format task name
                display_name = task_name.replace('_', ' ').title()
                if task_name == "overall_score":
                    display_name = "**Overall Score**"
                
                table_data.append({
                    "Metric": display_name,
                    "Base Model": f"{base_score:.3f}",
                    "Fine-tuned": f"{finetuned_score:.3f}",
                    "Improvement": f"{improvement:+.1%}",
                    "Status": "🟢" if improvement > 0 else "🔴" if improvement < 0 else "🟡"
                })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("", width="small"),
                "Improvement": st.column_config.TextColumn("Improvement", width="medium")
            }
        )
    
    def _render_performance_charts(self, comparison: ModelComparison) -> None:
        """Render performance visualization charts."""
        
        st.markdown("### 📈 Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            self._render_bar_chart(comparison)
        
        with col2:
            # Improvement radar chart
            self._render_radar_chart(comparison)
    
    def _render_bar_chart(self, comparison: ModelComparison) -> None:
        """Render bar chart comparison."""
        
        # Prepare data
        tasks = []
        base_scores = []
        finetuned_scores = []
        
        for task_name in comparison.base_scores:
            if task_name != "overall_score" and task_name in comparison.finetuned_scores:
                tasks.append(task_name.replace('_', ' ').title())
                base_scores.append(comparison.base_scores[task_name])
                finetuned_scores.append(comparison.finetuned_scores[task_name])
        
        # Create DataFrame for plotting
        chart_data = pd.DataFrame({
            "Task": tasks * 2,
            "Score": base_scores + finetuned_scores,
            "Model": ["Base Model"] * len(tasks) + ["Fine-tuned"] * len(tasks)
        })
        
        # Create bar chart
        fig = px.bar(
            chart_data,
            x="Task",
            y="Score",
            color="Model",
            barmode="group",
            title="Performance Comparison by Task",
            color_discrete_map={
                "Base Model": "#ff7f0e",
                "Fine-tuned": "#2ca02c"
            }
        )
        
        fig.update_layout(
            yaxis_title="Score",
            xaxis_title="Task",
            legend_title="Model Type",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_radar_chart(self, comparison: ModelComparison) -> None:
        """Render radar chart for improvements."""
        
        # Prepare data
        tasks = []
        improvements = []
        
        for task_name in comparison.improvements:
            if task_name != "overall_score":
                tasks.append(task_name.replace('_', ' ').title())
                improvements.append(comparison.improvements[task_name] * 100)  # Convert to percentage
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=improvements,
            theta=tasks,
            fill='toself',
            name='Improvement %',
            line_color='rgb(46, 160, 44)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min(improvements + [0]) - 5, max(improvements + [0]) + 5]
                )
            ),
            title="Improvement by Task (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_training_metrics(self, comparison: ModelComparison) -> None:
        """Render training-related metrics."""
        
        st.markdown("### ⚡ Training Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Training Duration**
            
            🕐 {comparison.training_time_hours:.1f} hours
            
            ⚡ {comparison.power_consumption_kwh:.1f} kWh consumed
            """)
        
        with col2:
            # Calculate efficiency metrics
            improvement_per_kwh = comparison.overall_improvement / comparison.power_consumption_kwh if comparison.power_consumption_kwh > 0 else 0
            
            st.info(f"""
            **Efficiency Metrics**
            
            📈 {comparison.overall_improvement:.1%} total improvement
            
            ⚡ {improvement_per_kwh:.2%} improvement per kWh
            """)
        
        with col3:
            # Environmental impact (rough estimate)
            co2_kg = comparison.power_consumption_kwh * 0.5  # Rough estimate: 0.5 kg CO2 per kWh
            
            st.info(f"""
            **Environmental Impact**
            
            🌍 ~{co2_kg:.1f} kg CO₂ equivalent
            
            🔋 {comparison.power_consumption_kwh:.1f} kWh total energy
            """)
    
    def _render_roi_analysis(self, comparison: ModelComparison) -> None:
        """Render ROI analysis."""
        
        st.markdown("### 💰 Return on Investment Analysis")
        
        # Calculate various ROI metrics
        training_hours = comparison.training_time_hours
        power_kwh = comparison.power_consumption_kwh
        improvement = comparison.overall_improvement
        
        # Estimated costs (rough estimates)
        electricity_cost = power_kwh * 0.12  # $0.12 per kWh
        hardware_depreciation = training_hours * 2.0  # $2 per hour hardware cost
        total_cost = electricity_cost + hardware_depreciation
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💵 Cost Breakdown")
            
            cost_data = pd.DataFrame({
                "Cost Type": ["Electricity", "Hardware Depreciation", "Total"],
                "Amount ($)": [electricity_cost, hardware_depreciation, total_cost]
            })
            
            fig = px.pie(
                cost_data[cost_data["Cost Type"] != "Total"],
                values="Amount ($)",
                names="Cost Type",
                title="Training Cost Breakdown"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 ROI Summary")
            
            # ROI metrics
            roi_metrics = {
                "Performance Gain": f"{improvement:.1%}",
                "Training Cost": f"${total_cost:.2f}",
                "Cost per % Improvement": f"${total_cost / (improvement * 100):.2f}" if improvement > 0 else "N/A",
                "Break-even Point": "Immediate" if improvement > 0 else "Never"
            }
            
            for metric, value in roi_metrics.items():
                st.metric(metric, value)
        
        # ROI recommendation
        if improvement > 0.1:  # 10% improvement
            st.success("🎉 **Excellent ROI!** Fine-tuning shows significant performance gains.")
        elif improvement > 0.05:  # 5% improvement
            st.info("👍 **Good ROI.** Fine-tuning provides meaningful improvements.")
        elif improvement > 0:
            st.warning("⚠️ **Marginal ROI.** Consider if the improvement justifies the cost.")
        else:
            st.error("❌ **Negative ROI.** Fine-tuning did not improve performance.")
    
    def render_summary_card(self, comparison: ModelComparison) -> None:
        """Render a compact summary card."""
        
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background-color: #f9f9f9;
            ">
                <h4>🔄 {comparison.base_model_id.split('/')[-1]} Comparison</h4>
                <p><strong>Overall Improvement:</strong> {comparison.overall_improvement:.1%}</p>
                <p><strong>Training Time:</strong> {comparison.training_time_hours:.1f} hours</p>
                <p><strong>Power Usage:</strong> {comparison.power_consumption_kwh:.1f} kWh</p>
                <p><strong>Date:</strong> {comparison.comparison_date.strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)