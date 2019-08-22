import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { gruvboxDark, solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';

import { GeneralOptions, UtilityOptions } from "../Options"

const styles = {
  card: {
    maxWidth: "100%",
  },
};

function DocsCard(props) {
  const { classes } = props;
  return (
    <Card className={classes.card}>
      <CardContent>
        <Typography gutterBottom variant="h3" component="h1">
          Branch
        </Typography>
        <Typography component="p">
          Create a new branch for a key
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'BRANCH -k <key> -b <new_branch> [-c <base_branch> | -u <refer_version>]'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._b}
        {GeneralOptions._c}
        {GeneralOptions._u}
        <Typography component="p">
          Utility Options:
        </Typography>
        {UtilityOptions._none}
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
ustore> branch -k myfirstKey -c master -b from-master
[SUCCESS: BRANCH] Branch "from-master" has been created for Key "myfirstKey"
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
