/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

/*Jacky start*/
#define N_STATES 48 //39?
#define MAX_SENTENCE 1000
#define INPUT_STATES 69
#define LLONG_MIN -9223372036854775807
/*Jacky end*/

#include <stdio.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  long     n;       /* number of examples */
  FILE *fp = NULL; 
  
  n=4000; /* replace by appropriate number of examples */
  examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE)*n);

  /* fill in your code here */
  char *sentID = (char*)my_malloc(sizeof(char)*100);//for sentence Id
  //sentID = "";
  int i=-1, j=0, k=0; //for iteration
  int frame_size = 0;
  int example_num = 0;
  //int sent_frame_size[3502] = {0};
  if ( (fp = fopen(file, "r")) == NULL) 
  {
	  perror("read example file failed!\n");
	  exit(0);
  }
  fscanf(fp, "%d", &example_num);
  int *sent_frame_size = malloc(sizeof(int)*example_num);
  for(i=0 ; i<example_num ; i++)
	  sent_frame_size[i] = 0;
  for(i=0 ; i<example_num ; i++)
  {
  	fscanf(fp, "%d", &sent_frame_size[i]);
  }
  i=0;
  int temp = 0;
  for(i=0 ; !feof(fp) && i < example_num ; i++)
  {
	frame_size = sent_frame_size[i];
	examples[i].x.len = frame_size;
	examples[i].y.len = frame_size;
	examples[i].x.seq = (float*)my_malloc(sizeof(float)*69*frame_size);
	examples[i].y.lab = (int*)my_malloc(sizeof(int)*frame_size);
	j=0;
   //printf("i: %d\n",i);
   //printf("frame: %d\n", frame_size);
	for(k=0 ; k<frame_size ; k++)
	{
   //printf("k: %d\n",k);
   		bzero(sentID, 100);
   //printf("fuck\n");
		fscanf(fp, "%s", sentID);
   //printf("fuck\n");
   //printf("frame_name: %s\n",sentID);
		fscanf(fp, "%d", &examples[i].y.lab[k]);
		for(temp=j ; temp < j+69 ; temp++)
		{	
			fscanf(fp, "%f", &examples[i].x.seq[j]);
   //if(i==3474 && k==29)printf(" %f", examples[i].x.seq[j]);
		}
		j = temp;
	}
  }
  fclose(fp);
  sample.n=n;
  sample.examples=examples;
  printf("example_num: %d\n", example_num);
  //printf("");
  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */

  sm->sizePsi = 48*69 + 48*48;
  sm->w = (double*)malloc(sizeof(double)*sm->sizePsi);
  MODEL * svm_model = (MODEL*)malloc(sizeof(MODEL));
  svm_model->kernel_parm.kernel_type = -1;
  sm->svm_model = svm_model;
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
  return(c);
}
int empty_label(LABEL y)
{
	return(0);
}

double eval_prob(float *x,double *w,int y,int frame)
{
  double prob=0;
  int i;
  for(i=0;i<INPUT_STATES;++i)
    prob+=w[y*INPUT_STATES+i+1]*x[frame*INPUT_STATES+i]; // The index to the vector w starts at 1, not at 0!
  return prob;
}
double getloss(int x,int y)
{
  return (x==y)? 0:1;
}


LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
  LABEL y;
  /*Jacky start*/
  double *prob_origin=(double*)malloc(sizeof(double)*N_STATES);
  double *prob_current=(double*)malloc(sizeof(double)*N_STATES);
  double *tmp_flip;
  int **max_track=(int **)malloc(sizeof(int*)*x.len);
  int i=0,j=0,k=0,max_lab=0;
  double prob_max=LLONG_MIN;
  /*prob init*/
  for(i=0;i<x.len-1;++i)
    max_track[i]=(int *)malloc(sizeof(int)*N_STATES);
  /*evaluate probability*/
  for(i=0;i<N_STATES;++i)
    prob_origin[i] = eval_prob(x.seq,sm->w,i,0);
  for(k=1;k<x.len;++k)
  {
    for(i=0;i<N_STATES;++i)
    {
      prob_current[i] =LLONG_MIN;
      for(j=0;j<N_STATES;++j)
      {
        double tmp=prob_origin[j]+sm->w[INPUT_STATES*N_STATES+j*N_STATES+i+1]+eval_prob(x.seq,sm->w,i,k);
        if(tmp > prob_current[i])
        {
            max_track[k-1][i] = j;
            prob_current[i] = tmp;
        }
      }
    }
    tmp_flip = prob_origin;
    prob_origin = prob_current;
    prob_current=  tmp_flip;
  }
  /*y init*/
  y.len=x.len;
  y.lab=(char*) malloc(sizeof(char)*x.len);
  for(i=0;i<N_STATES;++i)
  {
    if(prob_origin[i]>prob_max)
    {
      max_lab=i;
      prob_max =prob_origin[i];
    }
  }
  y.lab[y.len-1]=(char)max_lab;
  for(i=y.len-2;i>=0;--i)
  {
    max_lab=max_track[i][max_lab];
    y.lab[i]=(char)max_lab;
  }
  /*free malloc*/
  for(i=0;i<x.len-1;++i)
    free(max_track[i]);
  free(max_track);
  free(prob_origin);
  free(prob_current);
/*Jacky end*/
  /* insert your code for computing the predicted label y here */

  return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar)) 
     
     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
	LABEL ybar;

  /* insert your code for computing the label ybar here */
	
	//return(ybar);
	return(ybar);
}


/*Jacky start*/
LABEL find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
                 STRUCTMODEL *sm, 
                 STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)+psi(x,ybar)

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;
  /* insert your code for computing the label ybar here */
  /*Jacky start*/
  
  static int first=1;
  static double *prob_origin,*prob_current;
  static int **max_track;
  int i=0,k=0,j=0,max_lab = 0;
  double *tmp_flip,prob;
  double prob_max = LLONG_MIN;
  /*prob init*/
  
  if(first)
  {
    first=0;
    prob_origin=(double*)malloc(sizeof(double)*N_STATES);
    prob_current=(double*)malloc(sizeof(double)*N_STATES);
    max_track=(int **)malloc(sizeof(int *)*MAX_SENTENCE);
    for(i=0;i<MAX_SENTENCE-1;++i)
        max_track[i]=(int*)malloc(sizeof(int)*N_STATES);
  }
  /*evaluate probability*/
  for(k=0;k<x.len;++k)
  {
    for(i=0;i<N_STATES;++i)
    {
      if(k==0)   // first time 
        prob_origin[i]=eval_prob(x.seq,sm->w,i,k)+getloss(y.lab[0],k);
      else
      {
        prob_current[i]=LLONG_MIN;
        prob=eval_prob(x.seq,sm->w,i,k);
        for(j=0;j<N_STATES;++j) 
        {
          double tmp=prob_origin[j]+sm->w[INPUT_STATES*N_STATES+j*N_STATES+i+1]+prob+getloss(y.lab[k],i);
          if(tmp>prob_current[i])
          {
            max_track[k-1][i]=j;
            prob_current[i]=tmp;
          }
        }
      }
    }
    tmp_flip=prob_origin;
    prob_origin=prob_current;
    prob_current=tmp_flip;   //  why do this?
  }
  /*y bar init*/
  ybar.len=x.len;
  ybar.lab=(char*)malloc(sizeof(char)*x.len);
  for(i=0;i<N_STATES;++i) 
  {
    if(prob_origin[i]>prob_max)
    {
      prob_max=prob_origin[i];
      max_lab=i;
    }
  }
  ybar.lab[ybar.len-1] = (char)max_lab;
  for(i=ybar.len-2;i>-1;--i)  /*should modify*/
  {
    max_lab=max_track[i][max_lab];
    ybar.lab[i]=(char)max_lab;
  }
  /*Jacky end*/
  return(ybar);
}
/*Jacky end*/

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  SVECTOR *fvec = (SVECTOR*)malloc(sizeof(SVECTOR));
  WORD* temp = (WORD*)malloc(sizeof(WORD)*((69*48 + 48*48) + 1));
  fvec->words = temp;
  //fvec->twonorm_sq = 0.0;
  fvec->userdefined = (char*)malloc(sizeof(char));
  fvec->userdefined[0] = 0;
  //fvec->kernel_id = 0;
  fvec->next = NULL;
  fvec->factor = 1;
  int i=0, j=0, prev=-1, start, base= 48*69;
  double table[48*69 + 48*48] = {0.0};
  for(i=0 ; i<48*69 + 48*48 ; i++)
  {
	  fvec->words[i].wnum = i+1;
	  fvec->words[i].weight = 0.0f;
  }//initalization
  for(i=0 ; i<y.len ; i++)
  {
	  start = y.lab[i]*69;
	  for(j=0 ; j<69 ; j++)
	  {
		  table[start+j]+= x.seq[i*69 + j];
	  }
	  if(prev>=0)	table[base + 48*prev + y.lab[i]] += 1;
	  prev = y.lab[i];
  }
  for(i=0 ; i<48*69 + 48*48 ; i++)
  {
	  fvec->words[i].weight = table[i];
  }
  /* insert code for computing the feature vector for x and y here */
  return(fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
	double score = 0.0;
	int i=0;
	for(i=0 ; i<y.len ; i++)
	{
		score = score + (y.lab[i] == ybar.lab[i])? 0:1;
	}
	return score;
  if(sparm->loss_function == 0) { /* type 0 loss: 0/1 loss */
                                  /* return 0, if y==ybar. return 1 else */
  }
  else {
    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
  }
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
  }
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
	FILE *fp = fopen(file, "w");
	int i=0;
	for(i=0 ; i<sm->sizePsi ; i++)
	{
		fprintf(fp, "%f ", sm->w[i]);
	}
	fclose(fp);
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
	STRUCTMODEL ssvm;
	ssvm.sizePsi = 48*69 + 48*48;
	ssvm.w = (double*)malloc(sizeof(double)*ssvm.sizePsi);
	MODEL *svm_model = (MODEL*)malloc(sizeof(MODEL) );
	svm_model->kernel_parm.kernel_type = -1;
	ssvm.svm_model = svm_model;
	FILE *fp = fopen(file, "r");
	int i=0;
	for(i=0 ; i< ssvm.sizePsi ; i++)
	{
		fscanf(fp, "%f", &ssvm.w[i]);
	}
	fclose(fp);
	return ssvm;
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
	int i=0;
	for(i=0 ; i<y.len ; i++)
		fprintf(fp, "%d ", y.lab[i]);
	fprintf(fp, "\n");
} 

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
  free(x.seq);
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++)
  { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

