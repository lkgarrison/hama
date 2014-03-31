package org.apache.hama.graph;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.HashPartitioner;
import org.apache.hama.bsp.SequenceFileInputFormat;
import org.apache.hama.bsp.TextOutputFormat;
import org.apache.hama.commons.io.TextArrayWritable;

public class DegreeDistribution {

	public static class DegreeVertex extends
			Vertex<Text, NullWritable, IntWritable> {

		@Override
		public void compute(Iterable<IntWritable> messages) throws IOException {
			if (this.getSuperstepCount() == 0) {
				this.setValue(new IntWritable(getEdges().size()));
				sendMessageToNeighbors(new IntWritable(1));
			} else if (this.getSuperstepCount() == 1) {
				int sum = 0;
				for (IntWritable msg : messages) {
					sum += msg.get();
				}
				this.setValue(new IntWritable(this.getValue().get() + sum));
			} else {
				this.voteToHalt();
			}
		}
	}

	public static class Map extends MapReduceBase implements
			Mapper<LongWritable, Text, Text, IntWritable> {
		private final static IntWritable one = new IntWritable(1);

		public void map(LongWritable key, Text value,
				OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {
			output.collect(new Text(value.toString().split("\t")[1]), one);
		}
	}

	public static class Reduce extends MapReduceBase implements
			Reducer<Text, IntWritable, Text, IntWritable> {
		public void reduce(Text key, Iterator<IntWritable> values,
				OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {
			int sum = 0;
			while (values.hasNext()) {
				sum += values.next().get();
			}
			output.collect(key, new IntWritable(sum));
		}
	}

	public static class DegreeInputReader
			extends
			VertexInputReader<Text, TextArrayWritable, Text, NullWritable, IntWritable> {
		@Override
		public boolean parseVertex(Text key, TextArrayWritable value,
				Vertex<Text, NullWritable, IntWritable> vertex)
				throws Exception {
			vertex.setVertexID(key);
			for (Writable v : value.get()) {
				vertex.addEdge(new Edge<Text, NullWritable>((Text) v, null));
			}
			return true;
		}
	}

	public static GraphJob createDegreeJob(String[] args) throws IOException {
		HamaConfiguration conf = new HamaConfiguration(new Configuration());
		GraphJob degreeJob = new GraphJob(conf, DegreeDistribution.class);
		degreeJob.setJobName("DegreeGraph");
		degreeJob.setVertexClass(DegreeVertex.class);
		degreeJob.setInputPath(new Path(args[0]));
		degreeJob.setOutputPath(new Path(args[1]));
		if (args.length == 4) {
			degreeJob.setNumBspTask(Integer.parseInt(args[3]));
		}
		degreeJob.setAggregatorClass(AverageAggregator.class);
		degreeJob.setVertexInputReaderClass(DegreeInputReader.class);
		degreeJob.setVertexIDClass(Text.class);
		degreeJob.setVertexValueClass(IntWritable.class);
		degreeJob.setEdgeValueClass(NullWritable.class);
		degreeJob.setInputFormat(SequenceFileInputFormat.class);
		degreeJob.setPartitioner(HashPartitioner.class);
		degreeJob.setOutputFormat(TextOutputFormat.class);
		degreeJob.setOutputKeyClass(Text.class);
		degreeJob.setOutputValueClass(IntWritable.class);
		return degreeJob;
	}

	public static JobConf createDistributionJob(String[] args)
			throws IOException {
		JobConf conf = new JobConf(DegreeDistribution.class);
		conf.setJobName("DistributionMapReduce");
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(IntWritable.class);
		conf.setMapperClass(Map.class);
		conf.setCombinerClass(Reduce.class);
		conf.setReducerClass(Reduce.class);
		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(org.apache.hadoop.mapred.TextOutputFormat.class);
		FileInputFormat.setInputPaths(conf, new Path(args[1]));
		FileOutputFormat.setOutputPath(conf, new Path(args[2]));
		return conf;
	}

	private static void printUsage() {
		System.out
				.println("Usage: <input> <degree_output> <distribution_output> [tasks]");
		System.exit(-1);
	}

	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {
		if (args.length < 3)
			printUsage();
		long startTime = System.currentTimeMillis();
		GraphJob degreeJob = createDegreeJob(args);
		degreeJob.waitForCompletion(true);
		JobConf distributionJob = createDistributionJob(args);
		JobClient.runJob(distributionJob);
		System.out.println("Job Finished in "
				+ (System.currentTimeMillis() - startTime) / 1000.0
				+ " seconds");
	}
}
