def acc(output,target):
    output=output.argmax(dim=-1)
    return (output==target).sum().float()/target.size()[0]