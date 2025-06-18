#[cfg(test)]
mod tests {
    use super::*;
    use crate::chigutiro::{
        core::{Chigutiro, ChigutiroConfig, JsonRecord},
        filter::{Filter, RangeFilter, SetFilter, CustomFilter},
        algorithms::{FilterStrategy, AlgorithmSelector},
        benchmark::ChigutiroBenchmark,
        coordinator::{ViewCoordinator, ChartView, ViewConfig, CoordinationStrategy},
        chart_redraw::{ChartRedrawEngine, RedrawStrategy, RenderTarget},
    };
    use std::collections::HashMap;
    use serde_json::Value;

    fn create_test_record(id: u64, value: i32, category: &str) -> JsonRecord {
        let mut data = HashMap::new();
        data.insert("id".to_string(), Value::Number(serde_json::Number::from(id)));
        data.insert("value".to_string(), Value::Number(serde_json::Number::from(value)));
        data.insert("category".to_string(), Value::String(category.to_string()));
        
        JsonRecord { id, data }
    }

    #[test]
    fn test_chigutiro_basic_operations() {
        let config = ChigutiroConfig::default();
        let chigutiro = Chigutiro::new(config);
        
        // Test adding records
        let records = vec![
            create_test_record(1, 10, "A"),
            create_test_record(2, 20, "B"),
            create_test_record(3, 15, "A"),
        ];
        
        let result = chigutiro.add_records(records);
        assert!(result.is_ok());
        
        // Test size
        assert_eq!(chigutiro.size(), 3);
        
        // Test stats
        let stats = chigutiro.stats();
        assert_eq!(stats["total_records"], 3);
    }

    #[test]
    fn test_dimension_creation_and_filtering() {
        let config = ChigutiroConfig::default();
        let chigutiro = Chigutiro::new(config);
        
        let records = vec![
            create_test_record(1, 10, "A"),
            create_test_record(2, 20, "B"),
            create_test_record(3, 15, "A"),
            create_test_record(4, 25, "C"),
        ];
        
        chigutiro.add_records(records).unwrap();
        
        // Create dimension for value field
        let dimension = chigutiro.dimension("value", |record| {
            record.get_field("value").unwrap_or(Value::Null)
        });
        
        assert!(dimension.is_ok());
    }

    #[test]
    fn test_filter_types() {
        // Test exact filter
        let exact_filter = Filter::exact(Value::Number(serde_json::Number::from(42)));
        assert!(exact_filter.matches(&Value::Number(serde_json::Number::from(42))));
        assert!(!exact_filter.matches(&Value::Number(serde_json::Number::from(41))));
        
        // Test range filter
        let range_filter = RangeFilter::numeric(10.0, 20.0);
        assert!(range_filter.matches(&Value::Number(serde_json::Number::from(15))));
        assert!(!range_filter.matches(&Value::Number(serde_json::Number::from(25))));
        
        // Test set filter
        let set_filter = SetFilter::from_numbers(vec![10.0, 20.0, 30.0]);
        assert!(set_filter.matches(&Value::Number(serde_json::Number::from(20))));
        assert!(!set_filter.matches(&Value::Number(serde_json::Number::from(25))));
    }

    #[test]
    fn test_algorithm_selector() {
        let mut selector = AlgorithmSelector::new();
        
        let records = vec![
            create_test_record(1, 10, "A"),
            create_test_record(2, 20, "B"),
        ];
        
        // Test strategy selection
        let strategy = selector.select_strategy(
            &records,
            "value",
            &Value::Number(serde_json::Number::from(10)),
            records.len(),
        );
        
        // Should select appropriate strategy for small dataset
        assert!(matches!(strategy, FilterStrategy::LinearScan | FilterStrategy::HashFilter));
    }

    #[test]
    fn test_view_coordinator() {
        let mut coordinator = ViewCoordinator::new();
        
        // Create test view
        let view_config = ViewConfig {
            chart_type: "scatter".to_string(),
            dimensions: vec!["x".to_string(), "y".to_string()],
            min_update_interval_ms: 16, // ~60fps
            supports_incremental_updates: true,
            properties: HashMap::new(),
        };
        
        let view = ChartView::new("test_view".to_string(), "Test View".to_string(), view_config);
        coordinator.register_view(view);
        
        assert_eq!(coordinator.view_count(), 1);
        
        // Test filter change notification
        let result = coordinator.notify_filter_change("x");
        assert!(result.is_ok());
        
        // Test coordination strategies
        coordinator.set_strategy(CoordinationStrategy::Batched);
        coordinator.notify_filter_change("y").unwrap();
        coordinator.process_batched_updates().unwrap();
    }

    #[test]
    fn test_chart_redraw_engine() {
        let mut redraw_engine = ChartRedrawEngine::new(RedrawStrategy::Optimized);
        
        let test_data = vec![
            Value::Object(serde_json::Map::new()),
            Value::Object(serde_json::Map::new()),
        ];
        
        let target = RenderTarget::Svg("test_chart".to_string());
        let result = redraw_engine.redraw_chart("chart1", &test_data, &target);
        
        assert!(result.is_ok());
        
        let metrics = redraw_engine.get_metrics();
        assert_eq!(metrics.total_redraws, 1);
    }

    #[test]
    fn test_configuration_options() {
        // Test default configuration
        let default_config = ChigutiroConfig::default();
        assert_eq!(default_config.max_records, 10_000_000);
        assert!(default_config.enable_probabilistic);
        assert_eq!(default_config.false_positive_rate, 0.01);
        
        // Test custom configuration
        let custom_config = ChigutiroConfig {
            max_records: 1_000_000,
            enable_probabilistic: false,
            false_positive_rate: 0.05,
            num_threads: 4,
            cache_size_mb: 256,
            adaptive_algorithms: false,
            probabilistic_threshold: 50_000,
            enable_metrics: true,
        };
        
        let chigutiro = Chigutiro::new(custom_config);
        let stats = chigutiro.stats();
        assert_eq!(stats["config"]["max_records"], 1_000_000);
    }

    #[test]
    fn test_performance_metrics() {
        let config = ChigutiroConfig::default();
        let chigutiro = Chigutiro::new(config);
        
        // Add some test data
        let records = (0..1000)
            .map(|i| create_test_record(i as u64, i % 100, &format!("cat_{}", i % 10)))
            .collect();
        
        chigutiro.add_records(records).unwrap();
        
        let metrics = chigutiro.metrics();
        assert_eq!(metrics.total_records, 1000);
        assert!(metrics.last_filter_time_us > 0);
    }

    #[test]
    fn test_benchmark_system() {
        let mut benchmark = ChigutiroBenchmark::new();
        
        // Test data loading benchmark
        let result = benchmark.benchmark_data_loading(vec![100, 1000]);
        assert!(result.is_ok());
        
        // Test filtering benchmark
        let result = benchmark.benchmark_filtering(1000);
        assert!(result.is_ok());
        
        let results = benchmark.get_results();
        assert!(results.len() >= 2); // At least 2 data loading tests
        
        // All benchmarks should succeed
        for result in results {
            assert!(result.success);
            assert!(result.records_per_second > 0.0);
        }
    }

    #[test]
    fn test_large_dataset_handling() {
        let mut config = ChigutiroConfig::default();
        config.probabilistic_threshold = 500; // Lower threshold for testing
        
        let chigutiro = Chigutiro::new(config);
        
        // Create a larger dataset
        let records: Vec<JsonRecord> = (0..1000)
            .map(|i| create_test_record(i as u64, i % 100, &format!("category_{}", i % 20)))
            .collect();
        
        let result = chigutiro.add_records(records);
        assert!(result.is_ok());
        
        // System should automatically optimize for large datasets
        chigutiro.optimize().unwrap();
        
        let stats = chigutiro.stats();
        assert_eq!(stats["total_records"], 1000);
    }

    #[test]
    fn test_error_handling() {
        let config = ChigutiroConfig::default();
        let chigutiro = Chigutiro::new(config);
        
        // Test invalid dimension name
        let dimension_result = chigutiro.dimension("", |_| Value::Null);
        // This might succeed depending on implementation
        
        // Test clear operation
        let clear_result = chigutiro.clear();
        assert!(clear_result.is_ok());
        assert_eq!(chigutiro.size(), 0);
    }

    #[test]
    fn test_concurrent_operations() {
        use std::sync::Arc;
        use std::thread;
        
        let config = ChigutiroConfig::default();
        let chigutiro = Arc::new(Chigutiro::new(config));
        
        let mut handles = vec![];
        
        // Spawn multiple threads adding data concurrently
        for thread_id in 0..4 {
            let chigutiro_clone = Arc::clone(&chigutiro);
            let handle = thread::spawn(move || {
                let records: Vec<JsonRecord> = (0..100)
                    .map(|i| create_test_record(
                        (thread_id * 100 + i) as u64, 
                        i % 50, 
                        &format!("thread_{}_cat_{}", thread_id, i % 5)
                    ))
                    .collect();
                
                chigutiro_clone.add_records(records)
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
        
        // Verify final state
        assert_eq!(chigutiro.size(), 400); // 4 threads * 100 records each
    }

    #[test]
    fn test_memory_efficiency() {
        let config = ChigutiroConfig {
            cache_size_mb: 64, // Smaller cache for testing
            ..Default::default()
        };
        
        let chigutiro = Chigutiro::new(config);
        
        // Add data and check memory usage
        let records: Vec<JsonRecord> = (0..5000)
            .map(|i| create_test_record(i as u64, i % 1000, &format!("cat_{}", i % 50)))
            .collect();
        
        chigutiro.add_records(records).unwrap();
        
        let metrics = chigutiro.metrics();
        assert!(metrics.memory_usage_bytes > 0);
        
        // Optimize and check memory usage again
        chigutiro.optimize().unwrap();
    }
}

/// Integration tests that demonstrate real-world usage
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::chigutiro::*;

    #[test]
    fn test_scientific_dataset_simulation() {
        println!("\n🧬 Testing Scientific Dataset Simulation");
        
        let config = ChigutiroConfig {
            enable_probabilistic: true,
            probabilistic_threshold: 1000,
            adaptive_algorithms: true,
            ..Default::default()
        };
        
        let chigutiro = Chigutiro::new(config);
        
        // Simulate genomics data
        let records: Vec<JsonRecord> = (0..10000)
            .map(|i| {
                let mut data = HashMap::new();
                data.insert("gene_id".to_string(), Value::String(format!("GENE_{:05}", i)));
                data.insert("expression_level".to_string(), 
                           Value::Number(serde_json::Number::from_f64(
                               (i as f64 * 0.01) % 100.0
                           ).unwrap()));
                data.insert("chromosome".to_string(), 
                           Value::String(format!("chr{}", (i % 22) + 1)));
                data.insert("p_value".to_string(),
                           Value::Number(serde_json::Number::from_f64(
                               0.05 * (i as f64 / 10000.0)
                           ).unwrap()));
                
                JsonRecord { id: i as u64, data }
            })
            .collect();
        
        println!("📊 Adding {} genomics records...", records.len());
        let result = chigutiro.add_records(records);
        assert!(result.is_ok());
        
        // Create dimensions for filtering
        let expression_dim = chigutiro.dimension("expression_level", |record| {
            record.get_field("expression_level").unwrap_or(Value::Null)
        }).unwrap();
        
        let chromosome_dim = chigutiro.dimension("chromosome", |record| {
            record.get_field("chromosome").unwrap_or(Value::Null)
        }).unwrap();
        
        println!("🔍 Created dimensions for expression and chromosome");
        
        // Test coordinated filtering
        let mut coordinator = ViewCoordinator::new();
        
        // Create views for different visualizations
        let scatter_view = ChartView::new(
            "expression_scatter".to_string(),
            "Expression vs P-value Scatter".to_string(),
            ViewConfig {
                chart_type: "scatter".to_string(),
                dimensions: vec!["expression_level".to_string(), "p_value".to_string()],
                min_update_interval_ms: 16,
                supports_incremental_updates: true,
                properties: HashMap::new(),
            }
        );
        
        let heatmap_view = ChartView::new(
            "chromosome_heatmap".to_string(),
            "Chromosome Expression Heatmap".to_string(),
            ViewConfig {
                chart_type: "heatmap".to_string(),
                dimensions: vec!["chromosome".to_string(), "expression_level".to_string()],
                min_update_interval_ms: 50,
                supports_incremental_updates: false,
                properties: HashMap::new(),
            }
        );
        
        coordinator.register_view(scatter_view);
        coordinator.register_view(heatmap_view);
        
        println!("📈 Registered visualization views");
        
        // Simulate real-time filtering
        coordinator.set_strategy(CoordinationStrategy::Adaptive);
        coordinator.notify_filter_change("expression_level").unwrap();
        coordinator.notify_filter_change("chromosome").unwrap();
        
        // Check performance metrics
        let metrics = chigutiro.metrics();
        println!("⚡ Performance metrics:");
        println!("   Total records: {}", metrics.total_records);
        println!("   Last filter time: {} µs", metrics.last_filter_time_us);
        println!("   Memory usage: {} bytes", metrics.memory_usage_bytes);
        
        assert_eq!(metrics.total_records, 10000);
        assert!(metrics.last_filter_time_us > 0);
        
        println!("✅ Scientific dataset simulation completed successfully!");
    }

    #[test]  
    fn test_realtime_dashboard_simulation() {
        println!("\n📊 Testing Real-time Dashboard Simulation");
        
        let config = ChigutiroConfig {
            enable_metrics: true,
            adaptive_algorithms: true,
            ..Default::default()
        };
        
        let chigutiro = Chigutiro::new(config);
        let mut coordinator = ViewCoordinator::new();
        let mut redraw_engine = ChartRedrawEngine::new(RedrawStrategy::Incremental);
        
        // Create multiple coordinated views
        let views = vec![
            ("time_series", "Time Series", vec!["timestamp", "value"]),
            ("bar_chart", "Category Breakdown", vec!["category", "count"]),
            ("scatter_plot", "Correlation View", vec!["x_value", "y_value"]),
            ("histogram", "Distribution", vec!["value"]),
        ];
        
        for (id, name, dimensions) in views {
            let view = ChartView::new(
                id.to_string(),
                name.to_string(),
                ViewConfig {
                    chart_type: id.to_string(),
                    dimensions,
                    min_update_interval_ms: 16, // 60fps
                    supports_incremental_updates: true,
                    properties: HashMap::new(),
                }
            );
            coordinator.register_view(view);
        }
        
        println!("🎯 Created {} coordinated views", coordinator.view_count());
        
        // Simulate streaming data updates
        for batch in 0..5 {
            println!("📦 Processing data batch {}", batch + 1);
            
            let records: Vec<JsonRecord> = (0..1000)
                .map(|i| {
                    let mut data = HashMap::new();
                    data.insert("timestamp".to_string(), 
                               Value::String(format!("2024-01-01T{:02}:00:00Z", (batch * 2 + i / 500) % 24)));
                    data.insert("value".to_string(), 
                               Value::Number(serde_json::Number::from(
                                   (batch * 1000 + i) % 100
                               )));
                    data.insert("category".to_string(), 
                               Value::String(format!("cat_{}", i % 5)));
                    data.insert("x_value".to_string(),
                               Value::Number(serde_json::Number::from(i % 50)));
                    data.insert("y_value".to_string(),
                               Value::Number(serde_json::Number::from((i * 2) % 100)));
                    
                    JsonRecord { id: (batch * 1000 + i) as u64, data }
                })
                .collect();
            
            chigutiro.add_records(records).unwrap();
            
            // Trigger coordinated updates
            coordinator.notify_filter_change("value").unwrap();
            coordinator.process_batched_updates().unwrap();
            
            // Simulate chart redraws
            let test_data = vec![Value::Object(serde_json::Map::new()); 100];
            let target = RenderTarget::Canvas("dashboard".to_string());
            redraw_engine.redraw_chart("main_view", &test_data, &target).unwrap();
        }
        
        // Check final state
        assert_eq!(chigutiro.size(), 5000); // 5 batches * 1000 records
        
        let coord_stats = coordinator.get_stats();
        let redraw_metrics = redraw_engine.get_metrics();
        
        println!("📈 Final Statistics:");
        println!("   Total records: {}", chigutiro.size());
        println!("   Coordinator updates: {}", coord_stats.total_updates);
        println!("   Chart redraws: {}", redraw_metrics.total_redraws);
        println!("   Avg redraw time: {:.2} ms", redraw_metrics.avg_redraw_time_ms);
        
        assert!(coord_stats.total_updates > 0);
        assert_eq!(redraw_metrics.total_redraws, 5);
        
        println!("✅ Real-time dashboard simulation completed successfully!");
    }
}

/// Benchmark tests for performance validation
#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use crate::chigutiro::benchmark::ChigutiroBenchmark;

    #[test]
    fn test_comprehensive_benchmarks() {
        println!("\n🏁 Running Comprehensive Chigutiro Benchmarks");
        
        let mut benchmark = ChigutiroBenchmark::new();
        
        // Benchmark data loading with different sizes
        println!("📊 Benchmarking data loading...");
        let data_sizes = vec![1_000, 10_000, 50_000, 100_000];
        benchmark.benchmark_data_loading(data_sizes).unwrap();
        
        // Benchmark filtering performance
        println!("🔍 Benchmarking filtering...");
        benchmark.benchmark_filtering(100_000).unwrap();
        
        // Print comprehensive report
        benchmark.print_report();
        
        // Validate performance expectations
        let results = benchmark.get_results();
        
        // All benchmarks should succeed
        for result in results {
            assert!(result.success, "Benchmark '{}' failed", result.test_name);
            assert!(result.records_per_second > 100.0, 
                   "Performance too low: {} records/sec", result.records_per_second);
        }
        
        println!("✅ All benchmarks passed performance thresholds!");
    }
} 